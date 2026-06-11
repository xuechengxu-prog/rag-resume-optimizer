# RAG核心模块 —— 100% 适配 LangChain 1.2.18
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_qwen_llm(temperature=0.3):
    return ChatOpenAI(
        model="qwen3.6-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        max_tokens=4096
    )

# 向量库初始化 —— 只放入简历内容！
def init_vector_db(resume_chunks, jd_chunks):
    embedding = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    # 只使用简历chunk构建向量库，避免JD与JD匹配
    resume_texts = [chunk[1] for chunk in resume_chunks]
    vector_db = Chroma.from_texts(
        texts=resume_texts,
        embedding=embedding,
        persist_directory=None
    )
    return vector_db

# 匹配分析 —— 使用JD检索简历内容
def match_resume_jd(vector_db, jd_text):
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    llm = get_qwen_llm()

    template = """你是专业的求职顾问，根据提供的简历片段和岗位JD，完成以下分析：

## 任务要求：
1. 计算匹配度（0-100分）：根据简历内容与JD要求的符合程度打分
2. 分析依据：详细说明哪些要求满足，哪些不满足
3. 短板分析：列出简历中缺失或薄弱的技能/经验
4. 改进建议：针对短板给出具体的改进方向

## 输入信息：
JD要求：{question}

简历片段：{context}

## 输出格式：
请以结构化格式输出，包含匹配度评分、评分依据、短板分析和改进建议。"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n---\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(jd_text)

def optimize_resume(resume_text, jd_text, match_result):
    llm = get_qwen_llm(0.4)
    prompt = f"""你是专业的简历优化专家，请根据以下信息优化简历：

## 优化规则：
1. 不编造经历、不夸大事实，完全基于原始简历内容
2. 结合JD关键词优化表述
3. 使用STAR法则完善经历描述
4. 补充JD要求但简历缺失的关键词（自然融入）
5. 保持简历原有结构

## 输入信息：
原始简历：
{resume_text}

JD内容：
{jd_text}

匹配分析：
{match_result}

## 输出要求：
请直接输出优化后的完整简历，无需额外说明。"""
    return llm.invoke(prompt).content

def generate_report(match_result, optimized_resume, original_resume):
    llm = get_qwen_llm(0.3)
    prompt = f"""请基于以下信息生成简历改进建议报告：

## 报告结构：
1. 匹配度总结：复制匹配结果中的评分和依据
2. 核心短板及改进方向：分点说明
3. 优化亮点：对比原始简历说明改进之处

## 输入信息：
匹配结果：
{match_result}

优化后简历：
{optimized_resume}

原始简历：
{original_resume}

## 输出要求：
报告格式清晰，语言专业，便于求职者参考。"""
    return llm.invoke(prompt).content