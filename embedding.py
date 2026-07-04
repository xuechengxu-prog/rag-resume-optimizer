# 企业级嵌入模块
from langchain_community.embeddings import DashScopeEmbeddings
from config import config
import os


class EmbeddingManager:
    """嵌入向量管理器"""

    def __init__(self):
        self._embedding = None

    @property
    def embedding(self):
        if self._embedding is None:
            api_key = config.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
            self._embedding = DashScopeEmbeddings(
                model=config.EMBEDDING_MODEL,
                dashscope_api_key=api_key
            )
        return self._embedding

    def embed_documents(self, texts: list) -> list:
        """批量文档嵌入"""
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> list:
        """查询嵌入"""
        return self.embedding.embed_query(text)


embedding_manager = EmbeddingManager()
