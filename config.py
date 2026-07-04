# 企业级RAG配置管理
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM配置 - 从环境变量读取
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_MODEL = "qwen3.7-plus"
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 4096

    # Embedding配置
    EMBEDDING_MODEL = "text-embedding-v1"
    EMBEDDING_DIM = 1536

    # 向量数据库配置
    VECTOR_DB_PATH = "./chroma_db_enterprise"
    COLLECTION_NAME = "enterprise_rag"

    # 检索配置
    RETRIEVER_TOP_K = 10
    RETRIEVER_FETCH_K = 20
    MMR_LAMBDA = 0.55
    SCORE_THRESHOLD = 0.35
    RRF_K = 60

    # 分块配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    PARENT_CHUNK_SIZE = 2000
    PARENT_CHUNK_OVERLAP = 200

    # 评估配置
    EVALUATION_ENABLED = True
    FAITHFULNESS_THRESHOLD = 0.8
    RELEVANCY_THRESHOLD = 0.7

    # 文件上传配置
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".jpg", ".jpeg", ".png", ".md", ".docx"]

config = Config()
