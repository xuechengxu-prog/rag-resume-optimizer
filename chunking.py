# 企业级分块策略 —— 支持递归、语义、父子文档、Markdown结构感知
import hashlib
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from config import config


def create_chunk_metadata(
    text: str,
    chunk_index: int,
    parent_doc: str,
    doc_type: str = "resume",
    section: str = "",
    strategy: str = "recursive"
) -> dict:
    """创建chunk元数据模板"""
    chunk_id = hashlib.md5(
        f"{parent_doc}_{chunk_index}_{text[:50]}".encode()
    ).hexdigest()[:16]

    return {
        "chunk_id": chunk_id,
        "parent_doc": parent_doc,
        "section": section,
        "chunk_index": chunk_index,
        "word_count": len(text),
        "doc_type": doc_type,
        "strategy": strategy,
    }


class ChunkingEngine:
    """企业级分块引擎"""

    def __init__(self):
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,
            chunk_overlap=config.PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

    def recursive_chunk(
        self,
        text: str,
        parent_doc: str = "document",
        doc_type: str = "resume"
    ) -> List[Tuple[str, dict]]:
        """递归字符分块（推荐默认方案）"""
        docs = [Document(page_content=text)]
        chunks = self.recursive_splitter.split_documents(docs)

        result = []
        for i, chunk in enumerate(chunks):
            metadata = create_chunk_metadata(
                chunk.page_content, i, parent_doc, doc_type,
                strategy="recursive"
            )
            # 自动标签分类
            metadata["category"] = self._auto_categorize(chunk.page_content, doc_type)
            result.append((chunk.page_content, metadata))
        return result

    def parent_child_chunk(
        self,
        text: str,
        parent_doc: str = "document",
        doc_type: str = "resume"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]]]:
        """父子文档分块：小块检索，大块给LLM"""
        # 父块（大块）
        parent_docs = self.parent_splitter.split_documents(
            [Document(page_content=text)]
        )

        # 子块（小块）
        child_chunks = []
        parent_chunks = []

        for p_idx, parent in enumerate(parent_docs):
            parent_meta = create_chunk_metadata(
                parent.page_content, p_idx, parent_doc, doc_type,
                strategy="parent", section=f"parent_{p_idx}"
            )
            parent_chunks.append((parent.page_content, parent_meta))

            # 对父块继续切分为子块
            child_docs = self.child_splitter.split_documents(
                [Document(page_content=parent.page_content)]
            )

            for c_idx, child in enumerate(child_docs):
                child_meta = create_chunk_metadata(
                    child.page_content, f"{p_idx}_{c_idx}", parent_doc, doc_type,
                    strategy="child", section=f"child_of_parent_{p_idx}"
                )
                child_meta["parent_chunk_id"] = parent_meta["chunk_id"]
                child_chunks.append((child.page_content, child_meta))

        return child_chunks, parent_chunks

    def markdown_chunk(
        self,
        text: str,
        parent_doc: str = "document"
    ) -> List[Tuple[str, dict]]:
        """Markdown结构感知分块"""
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        try:
            chunks = splitter.split_text(text)
        except Exception:
            # 如果不是标准Markdown，回退到递归分块
            return self.recursive_chunk(text, parent_doc, doc_type="markdown")

        result = []
        for i, chunk in enumerate(chunks):
            metadata = create_chunk_metadata(
                chunk.page_content, i, parent_doc,
                doc_type="markdown", strategy="markdown"
            )
            # 合并文档原有元数据
            metadata.update(chunk.metadata)
            result.append((chunk.page_content, metadata))
        return result

    def semantic_chunk(
        self,
        text: str,
        parent_doc: str = "document",
        doc_type: str = "resume"
    ) -> List[Tuple[str, dict]]:
        """语义分块（使用嵌入模型计算相似度）"""
        # 先按句子切分
        sentences = re.split(r'([。！？；\n])', text)
        sentences = [
            (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")).strip()
            for i in range(0, len(sentences), 2)
            if sentences[i].strip()
        ]

        if len(sentences) <= 1:
            return self.recursive_chunk(text, parent_doc, doc_type)

        # 计算相邻句子相似度（使用简单字符重叠作为代理）
        chunks = []
        current_chunk = sentences[0]
        chunk_index = 0

        for i in range(1, len(sentences)):
            sim = self._char_overlap_ratio(current_chunk, sentences[i])
            # 如果相似度低，开始新chunk
            if sim < 0.3 and len(current_chunk) >= config.CHUNK_SIZE * 0.5:
                metadata = create_chunk_metadata(
                    current_chunk, chunk_index, parent_doc, doc_type,
                    strategy="semantic"
                )
                metadata["category"] = self._auto_categorize(current_chunk, doc_type)
                chunks.append((current_chunk, metadata))
                current_chunk = sentences[i]
                chunk_index += 1
            else:
                current_chunk += sentences[i]

        # 添加最后一个chunk
        if current_chunk:
            metadata = create_chunk_metadata(
                current_chunk, chunk_index, parent_doc, doc_type,
                strategy="semantic"
            )
            metadata["category"] = self._auto_categorize(current_chunk, doc_type)
            chunks.append((current_chunk, metadata))

        return chunks

    def _auto_categorize(self, text: str, doc_type: str) -> str:
        """自动分类chunk内容"""
        if doc_type == "resume":
            if any(kw in text for kw in ["技能", "熟练", "掌握", "精通", "熟悉", "Python", "Java"]):
                return "技能模块"
            elif any(kw in text for kw in ["项目", "负责", "参与", "完成", "开发", "设计", "实现"]):
                return "项目模块"
            elif any(kw in text for kw in ["工作", "经历", "实习", "任职", "职位", "公司"]):
                return "工作经历模块"
            elif any(kw in text for kw in ["教育", "学历", "学校", "专业", "学位", "本科", "硕士"]):
                return "教育背景模块"
            else:
                return "其他模块"
        elif doc_type == "jd":
            if any(kw in text for kw in ["要求", "职责", "工作内容", "负责", "职位"]):
                return "岗位要求模块"
            elif any(kw in text for kw in ["技能", "能力", "熟练", "掌握", "精通", "熟悉"]):
                return "技能要求模块"
            elif any(kw in text for kw in ["任职", "资格", "学历", "经验", "优先"]):
                return "任职资格模块"
            elif any(kw in text for kw in ["福利", "薪资", "待遇", "五险一金", "双休"]):
                return "福利待遇模块"
            else:
                return "其他模块"
        return "未分类"

    @staticmethod
    def _char_overlap_ratio(a: str, b: str) -> float:
        """计算两个字符串的字符重叠率（作为语义相似度的简单代理）"""
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        return len(intersection) / min(len(set_a), len(set_b))
