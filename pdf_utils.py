"""
PDF text extraction and preprocessing utilities.
"""
import pdfplumber
from typing import List, Optional
import io


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_file: Uploaded file object (Streamlit UploadedFile)
        
    Returns:
        Combined text from all pages of the PDF
    """
    text_content = []
    
    try:
        # Read PDF from bytes
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        return "\n\n".join(text_content)
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_from_multiple_pdfs(pdf_files: List) -> str:
    """
    Extract text from multiple PDF files and combine them.
    
    Args:
        pdf_files: List of uploaded PDF file objects
        
    Returns:
        Combined text from all PDFs
    """
    all_texts = []
    
    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)
            if text.strip():
                all_texts.append(f"--- Document: {pdf_file.name} ---\n{text}")
        except Exception as e:
            print(f"Warning: Could not extract text from {pdf_file.name}: {str(e)}")
            continue
    
    return "\n\n".join(all_texts)


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = ' '.join(line.split())
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


