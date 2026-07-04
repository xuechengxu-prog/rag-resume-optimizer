# 企业级检索模块 —— 混合检索 + RRF融合 + MMR重排
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from embedding import embedding_manager
from config import config


class HybridRetriever:
    """混合检索器：稠密向量 + 稀疏BM25 + RRF融合"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.vectorstore = None
        self.bm25 = None
        self.documents = []
        self.doc_texts = []
        self.doc_ids = []

    def build_index(self, chunks: List[Tuple[str, dict]]):
        """构建混合索引"""
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]

        # 构建稠密向量索引（Chroma）
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_manager.embedding,
            metadatas=metadatas,
            persist_directory=config.VECTOR_DB_PATH,
            collection_name=self.collection_name
        )

        # 构建稀疏BM25索引
        self.documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self.doc_texts = texts
        self.doc_ids = list(range(len(texts)))

        # 对中文进行简单分词（按字符）
        tokenized_corpus = [list(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """稠密向量检索"""
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        # 将Chroma结果映射到doc_id
        dense_results = []
        for doc, score in results:
            for idx, text in enumerate(self.doc_texts):
                if text == doc.page_content:
                    dense_results.append((idx, score))
                    break
        return dense_results

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """稀疏BM25检索"""
        if self.bm25 is None:
            return []

        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        sparse_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                sparse_results.append((int(idx), float(scores[idx])))
        return sparse_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        rrf_k: int = None,
        alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """混合检索 + RRF融合"""
        top_k = top_k or config.RETRIEVER_TOP_K
        rrf_k = rrf_k or config.RRF_K

        # 两路检索
        dense_results = self.dense_search(query, top_k=top_k * 2)
        sparse_results = self.sparse_search(query, top_k=top_k * 2)

        # 如果两路都为空，直接返回空
        if not dense_results and not sparse_results:
            return []

        # RRF融合
        scores = defaultdict(float)

        # 稠密结果按分数排序后赋予排名
        dense_sorted = sorted(dense_results, key=lambda x: -x[1])
        for rank, (doc_id, _) in enumerate(dense_sorted, start=1):
            scores[doc_id] += alpha * (1 / (rrf_k + rank))

        # 稀疏结果按分数排序后赋予排名
        sparse_sorted = sorted(sparse_results, key=lambda x: -x[1])
        for rank, (doc_id, _) in enumerate(sparse_sorted, start=1):
            scores[doc_id] += (1 - alpha) * (1 / (rrf_k + rank))

        # 按融合分数排序
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        # 转换为Document对象
        results = []
        for doc_id, score in sorted_results:
            doc = self.documents[doc_id]
            results.append((doc, score))

        return results


class MMRReranker:
    """MMR重排器：Maximal Marginal Relevance"""

    def __init__(
        self,
        lambda_mult: float = None,
    ):
        self.lambda_mult = lambda_mult or config.MMR_LAMBDA

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Document]:
        """MMR重排序（不对RRF分数做阈值过滤，因为RRF分数很小）"""
        if not candidates:
            return []

        # 如果候选数量不超过top_k，直接返回
        if len(candidates) <= top_k:
            return [doc for doc, _ in candidates]

        # 获取所有候选的embedding
        texts = [query] + [doc.page_content for doc, _ in candidates]
        try:
            embeddings = embedding_manager.embed_documents(texts)
            query_emb = np.array(embeddings[0])
            doc_embs = [np.array(emb) for emb in embeddings[1:]]
        except Exception:
            # embedding失败时直接按分数返回
            return [doc for doc, _ in candidates[:top_k]]

        # MMR贪心选择
        selected_indices = [0]  # 先选相关性最高的（candidates已按分数排序）
        remaining = list(range(1, len(candidates)))

        while len(selected_indices) < top_k and remaining:
            mmr_scores = []
            for idx in remaining:
                relevance = self._cosine_sim(query_emb, doc_embs[idx])
                # 计算与已选文档的最大相似度
                diversity = max([
                    self._cosine_sim(doc_embs[idx], doc_embs[sel_idx])
                    for sel_idx in selected_indices
                ]) if selected_indices else 0

                mmr_score = (
                    self.lambda_mult * relevance
                    - (1 - self.lambda_mult) * diversity
                )
                mmr_scores.append((idx, mmr_score))

            # 选择MMR分数最高的
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        # 返回选中的文档
        return [candidates[i][0] for i in selected_indices]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class EnterpriseRetriever:
    """企业级检索器：混合检索 + MMR重排"""

    def __init__(self):
        self.hybrid = HybridRetriever()
        self.reranker = MMRReranker()

    def build_index(self, chunks: List[Tuple[str, dict]]):
        """构建索引"""
        self.hybrid.build_index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_mmr: bool = True
    ) -> List[Document]:
        """检索文档"""
        top_k = top_k or config.RETRIEVER_TOP_K

        # 混合检索
        candidates = self.hybrid.hybrid_search(query, top_k=top_k * 2)

        if not candidates:
            return []

        if not use_mmr or len(candidates) <= top_k:
            return [doc for doc, _ in candidates[:top_k]]

        # MMR重排
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """检索文档并返回分数"""
        top_k = top_k or config.RETRIEVER_TOP_K
        return self.hybrid.hybrid_search(query, top_k=top_k)
