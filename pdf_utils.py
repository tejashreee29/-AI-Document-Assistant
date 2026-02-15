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
        
    Raises:
        Exception: If no text could be extracted from any PDF
    """
    if not pdf_files:
        raise ValueError("No PDF files provided")
    
    all_texts = []
    failed_files = []
    empty_files = []
    
    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)
            if text and text.strip():
                all_texts.append(f"--- Document: {pdf_file.name} ---\n{text}")
            else:
                empty_files.append(pdf_file.name)
                print(f"Warning: No text extracted from {pdf_file.name} - file may be empty or contain only images")
        except Exception as e:
            failed_files.append(pdf_file.name)
            print(f"Error: Could not process {pdf_file.name}: {str(e)}")
            continue
    
    # Provide helpful error messages
    if not all_texts:
        error_parts = ["Failed to extract text from any PDF files."]
        if failed_files:
            error_parts.append(f"Failed to process: {', '.join(failed_files)}")
        if empty_files:
            error_parts.append(f"Empty or image-only PDFs: {', '.join(empty_files)}")
        error_parts.append("Please ensure your PDFs contain extractable text (not just scanned images).")
        raise Exception(" ".join(error_parts))
    
    # Warn about partial failures
    if failed_files or empty_files:
        warning_parts = []
        if failed_files:
            warning_parts.append(f"Could not process: {', '.join(failed_files)}")
        if empty_files:
            warning_parts.append(f"No text in: {', '.join(empty_files)}")
        print(f"Warning: {' | '.join(warning_parts)}")
    
    combined_text = "\n\n".join(all_texts)
    
    # Final validation
    if len(combined_text.strip()) < 50:
        raise Exception(
            f"Extracted text is too short ({len(combined_text)} characters). "
            "Your PDFs may not contain enough readable text. "
            "Please ensure your documents have extractable text content."
        )
    
    return combined_text



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


