# 企业级文档解析模块 —— 支持多格式（PDF/图片/TXT/MD/DOCX）
import os
import re
from io import BytesIO
from typing import Optional

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.text import partition_text
    from unstructured.partition.md import partition_md
    from unstructured.partition.docx import partition_docx
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import easyocr
    import numpy as np
    from PIL import Image
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader


class EnterpriseDocumentParser:
    """企业级文档解析器，支持多格式、OCR、元数据提取"""

    def __init__(self):
        self._ocr_reader = None

    @property
    def ocr_reader(self):
        if self._ocr_reader is None and EASYOCR_AVAILABLE:
            self._ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        return self._ocr_reader

    def parse(self, file_path: str) -> dict:
        """解析文档，返回包含文本、元数据的字典"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return self._parse_image(file_path)
        elif ext == ".txt":
            return self._parse_txt(file_path)
        elif ext == ".md":
            return self._parse_md(file_path)
        elif ext == ".docx":
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _parse_pdf(self, file_path: str) -> dict:
        """解析PDF文档，优先使用PyMuPDF（中文支持最好），然后unstructured，最后PyPDF2"""
        text = ""
        metadata = {"source": file_path, "doc_type": "pdf", "pages": 0}

        # 第一选择：PyMuPDF（对中文PDF支持最佳）
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                page_texts = []
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        page_texts.append(page_text)
                text = "\n".join(page_texts)
                metadata["pages"] = len(doc)
                doc.close()
            except Exception as e:
                print(f"PyMuPDF解析失败: {e}")
                text = ""

        # 第二选择：unstructured
        if not text.strip() and UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_pdf(file_path, strategy="hi_res")
                text_parts = []
                for el in elements:
                    if hasattr(el, "text") and el.text:
                        text_parts.append(el.text)
                text = "\n".join(text_parts)
                metadata["pages"] = len(elements)
            except Exception as e:
                print(f"Unstructured PDF解析失败，降级到PyPDF2: {e}")
                text = ""

        # 第三选择：PyPDF2
        if not text.strip():
            reader = PdfReader(file_path)
            page_texts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text:
                    page_texts.append(page_text)
            text = "\n".join(page_texts)
            metadata["pages"] = len(reader.pages)

        return {"text": text, "metadata": metadata}

    def _parse_image(self, file_path: str) -> dict:
        """解析图片文档，使用EasyOCR进行中文OCR识别"""
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "图片解析需要EasyOCR。请安装: pip install easyocr"
            )

        img = np.array(Image.open(file_path))
        result = self.ocr_reader.readtext(img)

        text_lines = []
        for line in result:
            if line[1] and line[1].strip():
                text_lines.append(line[1])

        text = "\n".join(text_lines)
        if not text.strip():
            raise ValueError("图片中未识别到文本内容")

        metadata = {
            "source": file_path,
            "doc_type": "image",
            "ocr_boxes": len(result)
        }
        return {"text": text, "metadata": metadata}

    def _parse_txt(self, file_path: str) -> dict:
        """解析TXT文档"""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])

        metadata = {"source": file_path, "doc_type": "txt"}
        return {"text": text, "metadata": metadata}

    def _parse_md(self, file_path: str) -> dict:
        """解析Markdown文档"""
        if UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_md(file_path)
                text = "\n".join([el.text for el in elements if hasattr(el, "text")])
                metadata = {"source": file_path, "doc_type": "markdown"}
                return {"text": text, "metadata": metadata}
            except Exception:
                pass

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        metadata = {"source": file_path, "doc_type": "markdown"}
        return {"text": text, "metadata": metadata}

    def _parse_docx(self, file_path: str) -> dict:
        """解析DOCX文档"""
        if UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_docx(file_path)
                text = "\n".join([el.text for el in elements if hasattr(el, "text")])
                metadata = {"source": file_path, "doc_type": "docx"}
                return {"text": text, "metadata": metadata}
            except Exception:
                pass

        raise ValueError("DOCX解析需要unstructured库。请安装: pip install unstructured")


class DocumentCleaner:
    """企业级文档清洗流水线"""

    def __init__(self):
        self.ad_keywords = [
            "广告", "免费试用", "点击下载", "扫码关注", "版权所有",
            "转载", "未经许可", "禁止转载", "如有侵权请联系删除",
            "关注公众号", "微信扫码", "长按识别"
        ]

    def clean(self, text: str) -> str:
        """执行完整清洗流水线"""
        lines = text.split("\n")

        # 1. 检测并移除页眉页脚（出现在超过50%页面的短行）
        lines = self._remove_header_footer(lines)

        # 2. 过滤空白段落
        lines = [l for l in lines if len(l.strip()) >= 5]

        # 3. 过滤乱码（特殊字符占比过高）
        lines = [l for l in lines if not self._is_garbage(l)]

        # 4. 广告/水印关键词过滤
        lines = [l for l in lines if not any(kw in l for kw in self.ad_keywords)]

        # 5. 去重（连续重复段落）
        lines = self._remove_consecutive_duplicates(lines)

        # 6. 合并过短行（可能是PDF解析导致的断行）
        text = self._merge_short_lines(lines)

        # 7. 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def _remove_header_footer(self, lines: list) -> list:
        """移除页眉页脚"""
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        threshold = max(len(lines) * 0.05, 2)
        header_footer = {
            line for line, count in line_counts.items()
            if count > threshold
        }
        return [l for l in lines if l.strip() not in header_footer]

    def _is_garbage(self, line: str) -> bool:
        """检测乱码行"""
        special = sum(
            1 for c in line
            if not c.isalnum() and not "\u4e00" <= c <= "\u9fff"
            and c not in "，。！？；：\"\"''（）【】、\n\t "
        )
        return special / max(len(line), 1) > 0.5

    def _remove_consecutive_duplicates(self, lines: list) -> list:
        """移除连续重复行"""
        cleaned = []
        for line in lines:
            if not cleaned or cleaned[-1].strip() != line.strip():
                cleaned.append(line)
        return cleaned

    def _merge_short_lines(self, lines: list) -> str:
        """合并过短行（处理PDF断行问题）"""
        paragraphs = []
        current_para = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                continue

            # 如果行以标点结尾或者是标题，单独成段
            if stripped.endswith(("。", "！", "？", "；", ":", "：", ".", "!", "?"
            )) or stripped.startswith(("#", "##", "###", "【", "第")):
                current_para.append(stripped)
                paragraphs.append(" ".join(current_para))
                current_para = []
            else:
                current_para.append(stripped)

        if current_para:
            paragraphs.append(" ".join(current_para))

        return "\n\n".join(paragraphs)
