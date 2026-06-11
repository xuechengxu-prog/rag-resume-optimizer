#文本分块模块，优化检索精度

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 简历分块：按“技能、项目、经历”逻辑拆分，避免跨模块检索
def split_resume_text(resume_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 每个分块300字，适配简历片段长度
        chunk_overlap=50,  # 重叠50字，避免拆分丢失上下文
        separators=["\n\n", "\n", "，", "。"]  # 按中文标点拆分，更贴合中文习惯
    )
    chunks = text_splitter.split_text(resume_text)
    # 给每个分块添加标签，便于后续匹配（如“技能模块”“项目模块”）
    labeled_chunks = []
    for chunk in chunks:
        if any(keyword in chunk for keyword in ["技能", "熟练", "掌握", "精通"]):
            labeled_chunks.append(("技能模块", chunk))
        elif any(keyword in chunk for keyword in ["项目", "负责", "参与", "完成"]):
            labeled_chunks.append(("项目模块", chunk))
        else:
            labeled_chunks.append(("个人经历模块", chunk))
    return labeled_chunks

# JD分块：按“岗位要求、技能、任职资格”拆分
def split_jd_text(jd_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "，", "。"]
    )
    chunks = text_splitter.split_text(jd_text)
    labeled_chunks = []
    for chunk in chunks:
        if any(keyword in chunk for keyword in ["要求", "职责", "工作内容"]):
            labeled_chunks.append(("岗位要求模块", chunk))
        elif any(keyword in chunk for keyword in ["技能", "能力", "熟练", "掌握"]):
            labeled_chunks.append(("技能要求模块", chunk))
        elif any(keyword in chunk for keyword in ["任职", "资格", "学历", "经验"]):
            labeled_chunks.append(("任职资格模块", chunk))
        else:
            labeled_chunks.append(("其他模块", chunk))
    return labeled_chunks