# LLM客户端封装 —— 企业级RAG
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import config
import os


def get_qwen_llm(temperature=None, max_tokens=None):
    """获取Qwen LLM实例"""
    return ChatOpenAI(
        model=config.QWEN_MODEL,
        api_key=config.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY"),
        base_url=config.QWEN_BASE_URL,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
    )


def multi_query_rewrite(question: str, num_queries: int = 5) -> list:
    """Multi-Query改写：生成多个查询变体"""
    template = f"""你是一个AI语言模型助手。
你的任务是生成给定用户问题的{num_queries}个不同版本，用于从向量数据库中检索相关文档。
通过从多个角度生成用户问题，你的目标是帮助用户克服基于距离的相似度搜索的局限性。
请用换行符分隔这些替代问题。
原始问题是: {{question}}"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_qwen_llm(temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries[:num_queries]


def hyde_generate(question: str) -> str:
    """HyDE：生成假设性答案文档用于检索"""
    template = """请针对以下问题撰写一段专业的回答段落（即使你不确定确切答案）：
问题: {question}
段落:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_qwen_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


def step_back_rewrite(question: str) -> str:
    """Step-Back：生成更高层次的抽象问题"""
    template = """你是一个智能助手。给定一个具体的用户问题，请生成一个更抽象、更高层次的问题。
这个高层次问题应该能帮助回答原始问题。
原始问题: {question}
抽象问题:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_qwen_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})
