# Document loader module for resume and JD
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader
import os
import sys

# Read PDF file (common resume format)
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Extract text from each page, avoid None
    return text

# Read TXT file (common JD format)
def load_txt(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

# Check if EasyOCR is available
def is_easyocr_available():
    try:
        import easyocr
        return True
    except ImportError:
        return False

# Read image format JD (using EasyOCR for Chinese OCR)
def load_image(file_path):
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        
        # Initialize EasyOCR (Chinese)
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        
        # Read image
        img = np.array(Image.open(file_path))
        
        # Perform OCR
        result = reader.readtext(img)
        
        # Extract recognized text
        text_lines = []
        for line in result:
            if line[1] and line[1].strip():
                text_lines.append(line[1])
        
        text = '\n'.join(text_lines)
        
        if not text.strip():
            raise ValueError("No text content recognized in image")
        
        return text
        
    except ImportError:
        raise ImportError("Image format not supported! Please use:\n1. TXT format (recommended)\n2. PDF format\n\nTo use image recognition, install EasyOCR:\npip install easyocr")
    except Exception as e:
        raise ValueError(f"Image recognition failed: {str(e)}")

# Unified document loading entry, auto-detect file format
def load_document(file_path):
    if file_path.endswith(".pdf"):
        text = load_pdf(file_path)
        if not text.strip():
            raise ValueError("PDF file content is empty, cannot parse resume")
        return text
    elif file_path.endswith(".txt"):
        text = load_txt(file_path)
        if not text.strip():
            raise ValueError("TXT file content is empty")
        return text
    elif file_path.endswith((".jpg", ".png", ".jpeg")):
        text = load_image(file_path)
        if not text.strip():
            raise ValueError("Cannot recognize content from image file")
        return text
    else:
        raise ValueError("Unsupported file format, only PDF/TXT/JPG/PNG supported")