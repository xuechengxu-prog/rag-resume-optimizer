# 企业级简历优化 RAG 系统

基于 RAG（检索增强生成）技术的智能简历优化工具，从个人版升级为支持混合检索、MMR 重排、自动评估的企业级系统。

## 功能特性

### 核心功能
- **多格式文档解析**：支持 PDF、TXT、Markdown、DOCX、图片（JPG/PNG）
- **智能文档清洗**：自动去除页眉页脚、广告水印、乱码、重复段落
- **语义匹配分析**：基于向量数据库计算简历与 JD 的匹配度（0-100 分）
- **三级优化强度**：保守 / 标准 / 激进，灵活控制优化幅度
- **自动评估系统**：忠实度、JD 相关性、完整性、改进幅度、检索质量五维评估
- **多格式导出**：支持 PDF、DOCX 格式下载优化后的简历和报告

### 企业级 RAG 增强
- **混合检索**：稠密向量（Embedding）+ 稀疏检索（BM25）+ RRF 融合
- **MMR 重排**：Maximal Marginal Relevance，平衡相关性与多样性
- **多策略分块**：递归分块、父子文档分块、语义分块、Markdown 结构感知分块
- **Query 改写**：Multi-Query、HyDE、Step-Back 多种改写策略
- **自动分类标签**：按技能/项目/经历/教育等模块自动分类

## 技术架构

| 模块 | 技术 | 说明 |
|------|------|------|
| API 服务 | FastAPI + Uvicorn | 企业级后端服务 |
| Web 界面 | Streamlit / HTML 前端 | 双模式交互 |
| 文档解析 | PyMuPDF + Unstructured + EasyOCR | 多格式解析，中文优化 |
| 文档清洗 | 自定义清洗流水线 | 去页眉页脚/广告/乱码/断行合并 |
| 文本分块 | LangChain TextSplitter | 递归/父子/语义/Markdown 四种策略 |
| 向量数据库 | ChromaDB | 持久化存储，支持元数据过滤 |
| 混合检索 | Dense + BM25 + RRF | 稠密向量 + 稀疏检索融合 |
| 重排策略 | MMR | 最大边际相关性重排 |
| 嵌入模型 | DashScope Embeddings | text-embedding-v1，1536 维 |
| 大语言模型 | Qwen3.7-plus | 通过 DashScope API 调用 |
| 评估系统 | 基于 LLM 的轻量级评估 | 五维质量评估 + 综合评分 |

## 项目结构

```
rag-resume-optimizer/
├── api.py                 # FastAPI 后端服务（企业版入口）
├── app.py                 # Streamlit Web 应用（个人版入口）
├── rag_pipeline.py        # 企业级 RAG 流水线核心
├── config.py              # 统一配置管理
├── document_parser.py     # 多格式文档解析 + 清洗
├── chunking.py            # 智能分块引擎
├── retrieval.py           # 混合检索 + MMR 重排
├── llm_client.py          # LLM 客户端 + Query 改写
├── embedding.py           # 嵌入向量管理
├── evaluation.py          # RAG 质量评估
├── text_splitter.py       # 基础文本分块（兼容版）
├── document_loader.py     # 基础文档加载（兼容版）
├── rag_core.py            # 基础 RAG 核心（兼容版）
├── static/
│   └── index.html         # 前端页面
├── requirements.txt       # 项目依赖
├── .env.example           # 环境变量配置示例
├── start.bat              # Windows 启动脚本
├── start.ps1              # PowerShell 启动脚本
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境准备

- Python 3.9+
- [阿里云 DashScope API Key](https://dashscope.aliyun.com/)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 如需图片 OCR 功能，额外安装：`pip install easyocr`

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 运行方式（二选一）

**方式 A：FastAPI 企业版（推荐）**

```bash
# Windows
start.bat

# 或手动
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000 使用前端界面，或访问 http://localhost:8000/docs 查看 API 文档。

**方式 B：Streamlit 个人版**

```bash
streamlit run app.py
```

访问 http://localhost:8501。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/upload-resume` | POST | 上传简历文件 |
| `/api/upload-jd` | POST | 上传 JD 文件 |
| `/api/optimize` | POST | 执行简历优化 |
| `/api/full-pipeline` | POST | 完整流水线（上传+优化+评估） |
| `/api/evaluate` | POST | RAG 结果评估 |
| `/api/config` | GET | 获取系统配置 |
| `/api/download-resume` | POST | 下载优化后的简历 |
| `/api/download-report` | POST | 下载改进建议报告 |

## 核心流程

```
用户上传简历 + JD
    ↓
文档解析（PDF/DOCX/图片/TXT → 文本）
    ↓
文档清洗（去页眉页脚/广告/乱码）
    ↓
智能分块（递归/父子/语义/Markdown）
    ↓
构建混合索引（稠密向量 + BM25）
    ↓
JD 检索匹配（混合检索 + MMR 重排）
    ↓
LLM 智能优化（三级优化强度）
    ↓
生成改进报告
    ↓
自动质量评估（五维评分）
    ↓
展示结果 + 导出 PDF/DOCX
```

## 配置说明

在 `config.py` 中可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_MODEL` | qwen3.7-plus | 大语言模型 |
| `CHUNK_SIZE` | 500 | 分块大小 |
| `RETRIEVER_TOP_K` | 10 | 检索返回数量 |
| `MMR_LAMBDA` | 0.55 | MMR 相关性权重 |
| `MAX_FILE_SIZE` | 50MB | 上传文件大小限制 |

## 注意事项

- API Key 请妥善保管，不要上传到公开仓库
- 首次运行 EasyOCR 会自动下载模型，可能需要一些时间
- 确保 PDF 文件包含可识别的文本内容
- 企业版（FastAPI）和个人版（Streamlit）可以独立运行

## 许可证

MIT License
