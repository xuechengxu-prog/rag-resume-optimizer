#Web交互界面，Streamlit搭建，一键运行
import streamlit as st
from document_loader import load_document, is_easyocr_available
from text_splitter import split_resume_text, split_jd_text
from rag_core import init_vector_db, match_resume_jd, optimize_resume, generate_report
import os
from io import BytesIO

def generate_pdf(text):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    try:
        pdfmetrics.registerFont(TTFont('SimSun', r'C:\Windows\Fonts\simsun.ttc'))
        p.setFont('SimSun', 11)
        font_name = 'SimSun'
    except:
        try:
            pdfmetrics.registerFont(TTFont('SimHei', r'C:\Windows\Fonts\simhei.ttf'))
            p.setFont('SimHei', 11)
            font_name = 'SimHei'
        except:
            p.setFont('Helvetica', 11)
            font_name = 'Helvetica'
    
    text_lines = text.split('\n')
    y = height - 50
    line_height = 16
    left_margin = 50
    right_margin = 50
    
    for line in text_lines:
        if y < 50:
            p.showPage()
            if font_name != 'Helvetica':
                p.setFont(font_name, 11)
            else:
                p.setFont('Helvetica', 11)
            y = height - 50
        
        words = []
        current_line = ''
        for char in line:
            if p.stringWidth(current_line + char, font_name, 11) < (width - left_margin - right_margin):
                current_line += char
            else:
                if current_line:
                    words.append(current_line)
                current_line = char
        if current_line:
            words.append(current_line)
        
        for word in words:
            if y < 50:
                p.showPage()
                if font_name != 'Helvetica':
                    p.setFont(font_name, 11)
                else:
                    p.setFont('Helvetica', 11)
                y = height - 50
            p.drawString(left_margin, y, word)
            y -= line_height
    
    p.save()
    buffer.seek(0)
    return buffer.read()

# 设置页面标题和布局
st.set_page_config(page_title="智能简历优化RAG系统", page_icon="📄", layout="wide")
st.title("📄 智能简历优化RAG系统（JD匹配+自动优化）")

# 初始化session state
if "optimized_resume" not in st.session_state:
    st.session_state.optimized_resume = None
if "report" not in st.session_state:
    st.session_state.report = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

# 清空按钮
if st.button("清空结果"):
    st.session_state.optimized_resume = None
    st.session_state.report = None
    st.session_state.match_result = None
    st.session_state.resume_text = None
    st.rerun()

# 上传文件（左侧上传区）
with st.sidebar:
    st.header("上传文件")
    resume_file = st.file_uploader("上传你的简历（PDF格式）", type=["pdf"])
    
    # Check if EasyOCR is available and set supported formats
    if is_easyocr_available():
        jd_types = ["txt", "pdf", "jpg", "png", "jpeg"]
        jd_label = "上传目标岗位JD（TXT/PDF/JPG/PNG）"
    else:
        jd_types = ["txt", "pdf"]
        jd_label = "上传目标岗位JD（TXT/PDF）"
    
    jd_file = st.file_uploader(jd_label, type=jd_types)
    optimize_btn = st.button("一键优化", disabled=not (resume_file and jd_file))

# 核心逻辑执行（右侧展示区）
# 如果有保存的结果，先显示
if st.session_state.optimized_resume and st.session_state.report and st.session_state.match_result:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始简历")
        st.text_area("原始简历内容", st.session_state.resume_text, height=400)
    with col2:
        st.subheader("优化后简历")
        st.text_area("优化后简历内容", st.session_state.optimized_resume, height=400)
    
    st.subheader("📊 简历与JD匹配分析")
    st.write(st.session_state.match_result)
    st.subheader("📋 改进建议报告")
    st.write(st.session_state.report)
    
    # 导出功能 - 提供PDF下载
    pdf_content = generate_pdf(st.session_state.optimized_resume)
    st.download_button(
        "下载优化后简历(PDF)",
        pdf_content,
        file_name="优化后简历.pdf",
        mime="application/pdf"
    )
    st.download_button("下载改进建议报告", st.session_state.report, file_name="简历改进建议报告.txt")

elif optimize_btn and resume_file and jd_file:
    try:
        # 1. 保存上传的文件到本地（临时文件夹）
        os.makedirs("./temp", exist_ok=True)
        resume_path = f"./temp/{resume_file.name}"
        jd_path = f"./temp/{jd_file.name}"
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())
        with open(jd_path, "wb") as f:
            f.write(jd_file.getbuffer())
        
        # 2. 加载并解析文档
        st.info("正在解析简历和JD...")
        resume_text = load_document(resume_path)
        jd_text = load_document(jd_path)
        
        # 检查简历内容是否为空
        if not resume_text.strip():
            st.error("错误：简历文件内容为空，请确保上传的PDF文件包含可识别的文本内容")
            raise ValueError("简历内容为空")
        
        st.success(f"简历解析成功，共 {len(resume_text)} 字符")
        st.success(f"JD解析成功，共 {len(jd_text)} 字符")
        
        # 3. 文本分块
        st.info("正在进行文本分块，优化检索精度...")
        resume_chunks = split_resume_text(resume_text)
        jd_chunks = split_jd_text(jd_text)
        
        st.success(f"简历分块完成，共 {len(resume_chunks)} 块")
        st.success(f"JD分块完成，共 {len(jd_chunks)} 块")
        
        # 4. 初始化向量库，执行语义匹配
        st.info("正在进行JD与简历匹配，分析短板...")
        vector_db = init_vector_db(resume_chunks, jd_chunks)
        match_result = match_resume_jd(vector_db, jd_text)
        
        # 5. 智能优化简历，生成报告
        st.info("正在优化简历，生成改进报告...")
        optimized_resume = optimize_resume(resume_text, jd_text, match_result)
        report = generate_report(match_result, optimized_resume, resume_text)
        
        # 6. 展示结果（分栏展示，清晰直观）
        # 保存结果到session state
        st.session_state.optimized_resume = optimized_resume
        st.session_state.report = report
        st.session_state.match_result = match_result
        st.session_state.resume_text = resume_text
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始简历")
            st.text_area("原始简历内容", resume_text, height=400)
        with col2:
            st.subheader("优化后简历")
            st.text_area("优化后简历内容", optimized_resume, height=400)
        
        # 展示匹配结果和改进报告
        st.subheader("📊 简历与JD匹配分析")
        st.write(match_result)
        st.subheader("📋 改进建议报告")
        st.write(report)
        
        # 导出功能
        pdf_content = generate_pdf(optimized_resume)
        st.download_button(
            "下载优化后简历(PDF)",
            pdf_content,
            file_name="优化后简历.pdf",
            mime="application/pdf"
        )
        st.download_button("下载改进建议报告", report, file_name="简历改进建议报告.txt")
        
        # 删除临时文件，避免占用空间
        os.remove(resume_path)
        os.remove(jd_path)
        
    except Exception as e:
        st.error(f"处理过程中发生错误：{str(e)}")
        st.error("请检查上传的文件是否正确，或尝试重新上传")
        import traceback
        st.exception(e)