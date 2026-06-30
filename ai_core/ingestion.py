import fitz
import io
from PIL import Image
import pytesseract
from docx import Document
import pandas as pd
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_pdf(path):
    doc = fitz.open(path)

    text = ""

    for page in doc:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text
        else:
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(image)

    return text

def extract_docx(path):

    doc = Document(path)

    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return "\n".join(text)


def extract_excel(path):

    excel = pd.ExcelFile(path)

    text = ""

    for sheet in excel.sheet_names:

        df = pd.read_excel(path, sheet_name=sheet)

        text += df.to_string()

    return text


from PIL import Image

def extract_image(path):

    img = Image.open(path)

    return pytesseract.image_to_string(img)

def extract_csv(path):
    df = pd.read_csv(path)

    print(df.head())
    print(df.columns)
    print(df.shape)

    return df.head().to_string(index=False)

def ingest_document(path):

    if path.endswith(".pdf"):
        return extract_pdf(path)

    elif path.endswith(".docx"):
        return extract_docx(path)

    elif path.endswith(".xlsx"):
        return extract_excel(path)
    
    elif path.endswith(".csv"):
        return extract_csv(path)

    elif path.endswith(".png") or path.endswith(".jpg"):
        return extract_image(path)
    
print("Program started")

path = r"D:\hackathons\data\fake_news_data.csv"

print("File exists:", os.path.exists(path))

text = ingest_document(path)

print("Extraction complete")
print(repr(text))