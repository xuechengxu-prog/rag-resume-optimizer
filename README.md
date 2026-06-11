# 智能简历优化 RAG 系统

基于 RAG（检索增强生成）技术的智能简历优化工具，支持简历与岗位 JD 的智能匹配分析、自动优化和生成改进报告。

## 功能特性

- **智能文档解析**：支持 PDF、TXT、图片（JPG/PNG）格式的简历和 JD 解析
- **语义匹配分析**：基于向量数据库计算简历与 JD 的匹配度（0-100 分）
- **自动简历优化**：结合 JD 关键词和 STAR 法则优化简历表述
- **改进建议报告**：生成结构化的简历改进建议
- **PDF 导出**：支持下载优化后的简历为 PDF 格式
- **Web 交互界面**：基于 Streamlit 的友好操作界面

## 技术架构

| 模块 | 技术 | 说明 |
|------|------|------|
| Web 界面 | Streamlit | 一键运行的交互界面 |
| 文档解析 | PyPDF2 + EasyOCR | 支持 PDF/TXT/图片 |
| 文本分块 | LangChain TextSplitter | 按技能/项目/经历智能分块 |
| 向量数据库 | ChromaDB | 简历内容语义存储与检索 |
| 嵌入模型 | DashScope Embeddings | text-embedding-v1 |
| 大语言模型 | Qwen3.6-plus | 通过 DashScope API 调用 |

## 项目结构

```
rag-resume-optimizer/
├── app.py              # Streamlit Web 应用主入口
├── document_loader.py  # 文档加载与解析模块
├── text_splitter.py    # 智能文本分块模块
├── rag_core.py         # RAG 核心逻辑（向量库/匹配/优化）
├── requirements.txt    # 项目依赖
└── temp/               # 临时文件目录（自动创建）
```

## 快速开始

### 1. 环境准备

- Python 3.9+
- [阿里云 DashScope API Key](https://dashscope.aliyun.com/)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开（默认 http://localhost:8501）。

## 使用说明

1. **上传简历**：在左侧栏上传你的简历（PDF 格式）
2. **上传 JD**：上传目标岗位的 JD（支持 TXT/PDF/图片格式）
3. **一键优化**：点击"一键优化"按钮
4. **查看结果**：
   - 左侧显示原始简历
   - 右侧显示优化后的简历
   - 下方显示匹配度分析和改进建议报告
5. **导出文件**：下载优化后的 PDF 简历和改进建议报告

## 核心流程

```
用户上传简历 + JD
    ↓
文档解析（PDF/TXT/图片 → 文本）
    ↓
智能分块（按技能/项目/经历拆分）
    ↓
构建向量库（简历内容嵌入存储）
    ↓
JD 检索匹配（计算匹配度）
    ↓
LLM 智能优化（生成优化简历）
    ↓
生成改进报告
    ↓
展示结果 + 导出 PDF
```

## 依赖清单

- streamlit
- langchain
- langchain-community
- chromadb
- PyPDF2
- transformers
- pillow
- reportlab
- python-dotenv

## 注意事项

- 图片格式 JD 需要安装 EasyOCR：`pip install easyocr`
- 首次运行时会自动下载 OCR 模型，可能需要一些时间
- 确保 PDF 文件包含可识别的文本内容（非扫描版图片 PDF）
- API Key 请妥善保管，不要上传到公开仓库

## 许可证

MIT License
