# 企业级RAG流水线 —— 简历优化场景
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Optional
from document_parser import EnterpriseDocumentParser, DocumentCleaner
from chunking import ChunkingEngine
from retrieval import EnterpriseRetriever
from llm_client import get_qwen_llm, multi_query_rewrite, hyde_generate, step_back_rewrite
from config import config


# 全局Prompt格式约束
NO_MARKDOWN_RULE = """【重要格式要求】输出必须是纯文本格式，禁止使用任何Markdown标记符号，包括：
- 禁止使用的符号：# * > - ` 【】 ~~ || 等
- 标题直接写文字，不要加井号
- 列表项直接换行写，不要加短横线或星号
- 强调文字直接写，不要加星号或粗体标记
- 正常段落间距即可，段与段之间空一行"""


class ResumeOptimizationRAG:
    """简历优化企业级RAG系统"""

    def __init__(self):
        self.parser = EnterpriseDocumentParser()
        self.cleaner = DocumentCleaner()
        self.chunker = ChunkingEngine()
        self.retriever = EnterpriseRetriever()
        self.resume_text = ""
        self.jd_text = ""

    def ingest_resume(self, file_path: str) -> dict:
        """摄入简历文档"""
        parsed = self.parser.parse(file_path)
        cleaned = self.cleaner.clean(parsed["text"])
        self.resume_text = cleaned

        # 根据文档类型选择分块策略
        ext = file_path.split(".")[-1].lower()
        if ext == "md":
            chunks = self.chunker.markdown_chunk(cleaned, parent_doc="resume")
        else:
            chunks = self.chunker.recursive_chunk(
                cleaned, parent_doc="resume", doc_type="resume"
            )

        # 构建检索索引（仅使用简历内容）
        self.retriever.build_index(chunks)

        return {
            "text": cleaned,
            "metadata": parsed["metadata"],
            "chunks": len(chunks)
        }

    def ingest_jd(self, file_path: str) -> dict:
        """摄入JD文档"""
        parsed = self.parser.parse(file_path)
        cleaned = self.cleaner.clean(parsed["text"])
        self.jd_text = cleaned

        return {
            "text": cleaned,
            "metadata": parsed["metadata"]
        }

    def set_jd_text(self, text: str):
        """直接设置JD文本"""
        self.jd_text = self.cleaner.clean(text)

    def match_resume_jd(self) -> dict:
        """简历与JD匹配分析"""
        if not self.resume_text or not self.jd_text:
            raise ValueError("请先上传简历和JD")

        # 直接用JD原文检索简历内容
        docs = self.retriever.retrieve(self.jd_text, top_k=5, use_mmr=True)

        # 如果检索结果为空，回退到使用完整简历文本
        if docs:
            context = "\n---\n".join([d.page_content for d in docs[:10]])
        else:
            context = self.resume_text[:4000]

        # 生成匹配分析
        llm = get_qwen_llm(temperature=0.3, max_tokens=8192)

        template = f"""你是专业的求职顾问。请根据简历内容和岗位JD进行匹配分析。

{NO_MARKDOWN_RULE}

分析要点：
1. 匹配度评分（0-100分）
2. 满足的要求和未满足的要求
3. 简历中缺失或薄弱的技能/经验
4. 针对短板给出具体改进方向

JD要求：{{jd}}

简历内容：{{context}}

请直接输出分析结果，纯文本格式。"""

        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        match_result = chain.invoke({"jd": self.jd_text, "context": context})

        return {
            "match_result": match_result,
            "retrieved_chunks": len(docs)
        }

    def optimize_resume(
        self,
        match_result: str,
        optimization_level: str = "standard"
    ) -> str:
        """智能优化简历"""
        llm = get_qwen_llm(temperature=0.4, max_tokens=8192)

        level_instructions = {
            "conservative": "仅做最小化修改，保持原始表述的95%以上不变。",
            "standard": "在保持真实性的前提下，优化关键词匹配和表述清晰度。",
            "aggressive": "大幅重构简历结构和表述，最大化JD匹配度。"
        }

        prompt = f"""你是专业的简历优化专家。请根据原始简历和JD要求进行优化。

{NO_MARKDOWN_RULE}

优化规则：
1. 不编造经历、不夸大事实，完全基于原始简历内容
2. 结合JD关键词优化表述
3. 使用STAR法则完善经历描述
4. 补充JD要求但简历缺失的关键词（自然融入）
5. 保持简历原有结构
6. {level_instructions.get(optimization_level, level_instructions["standard"])}

原始简历：
{self.resume_text}

JD内容：
{self.jd_text}

匹配分析：
{match_result}

请直接输出优化后的完整简历，纯文本格式，无需额外说明。"""

        return llm.invoke(prompt).content

    def generate_report(
        self,
        match_result: str,
        optimized_resume: str
    ) -> str:
        """生成改进建议报告"""
        llm = get_qwen_llm(temperature=0.3, max_tokens=8192)

        prompt = f"""请基于以下信息生成简历改进建议报告。

{NO_MARKDOWN_RULE}

报告结构：
1. 匹配度总结
2. 核心短板及改进方向
3. 优化亮点（对比原始简历）
4. 求职建议

匹配结果：
{match_result}

优化后简历：
{optimized_resume}

原始简历：
{self.resume_text}

请直接输出报告，纯文本格式，语言专业。"""

        return llm.invoke(prompt).content

    def run_full_pipeline(
        self,
        resume_path: str,
        jd_path: str,
        optimization_level: str = "standard"
    ) -> dict:
        """运行完整RAG流水线"""
        # 1. 文档摄入
        resume_info = self.ingest_resume(resume_path)
        jd_info = self.ingest_jd(jd_path)

        # 2. 匹配分析
        match = self.match_resume_jd()

        # 3. 简历优化
        optimized = self.optimize_resume(
            match["match_result"],
            optimization_level=optimization_level
        )

        # 4. 生成报告
        report = self.generate_report(match["match_result"], optimized)

        return {
            "resume_info": resume_info,
            "jd_info": jd_info,
            "match_result": match["match_result"],
            "retrieved_chunks": match["retrieved_chunks"],
            "optimized_resume": optimized,
            "report": report
        }
