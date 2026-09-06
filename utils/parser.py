import PyPDF2
import docx

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text
