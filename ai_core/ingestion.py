import fitz
import io
import os
import logging
import pandas as pd
import pytesseract

from PIL import Image
from docx import Document

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ==========================================================
# Configure Tesseract
# ==========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================================================
# PDF Extraction
# ==========================================================

def extract_pdf(path):
    """
    Extract text from PDF.
    Uses OCR for scanned pages.
    """

    document = fitz.open(path)

    text = ""

    for page in document:

        page_text = page.get_text()

        if page_text.strip():

            text += page_text

        else:

            pix = page.get_pixmap(dpi=300)

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            text += pytesseract.image_to_string(image)

    return text


# ==========================================================
# DOCX Extraction
# ==========================================================

def extract_docx(path):
    """
    Extract text from Word document.
    """

    document = Document(path)

    text = []

    for paragraph in document.paragraphs:

        text.append(paragraph.text)

    return "\n".join(text)


# ==========================================================
# Excel Extraction
# ==========================================================

def extract_excel(path):
    """
    Extract all sheets from Excel.
    """

    excel = pd.ExcelFile(path)

    text = ""

    for sheet in excel.sheet_names:

        dataframe = pd.read_excel(
            path,
            sheet_name=sheet
        )

        text += dataframe.to_string()

        text += "\n\n"

    return text


# ==========================================================
# CSV Extraction
# ==========================================================

def extract_csv(path):
    """
    Extract CSV file.
    """

    dataframe = pd.read_csv(path)

    logging.info(f"Rows    : {len(dataframe)}")
    logging.info(f"Columns : {len(dataframe.columns)}")

    return dataframe.to_string(index=False)


# ==========================================================
# Image OCR
# ==========================================================

def extract_image(path):
    """
    Extract text from image using OCR.
    """

    image = Image.open(path)

    return pytesseract.image_to_string(image)


# ==========================================================
# NASA TXT Extraction
# ==========================================================

def extract_nasa_txt(path):
    """
    Read NASA C-MAPSS dataset.
    """

    logging.info(
        f"Reading NASA Dataset : {os.path.basename(path)}"
    )

    dataframe = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    logging.info(
        f"Rows    : {len(dataframe)}"
    )

    logging.info(
        f"Columns : {len(dataframe.columns)}"
    )

    return dataframe.to_string(index=False)


# ==========================================================
# Generic TXT Extraction
# ==========================================================

def extract_txt(path):

    # NASA dataset
    if "FD00" in os.path.basename(path):
        return extract_nasa_txt(path)

    with open(path, "r", encoding="latin-1") as file:
        return file.read()


# ==========================================================
# Universal Document Loader
# ==========================================================

def ingest_document(path):
    """
    Load any supported document.
    """

    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":

        return extract_pdf(path)

    elif extension == ".docx":

        return extract_docx(path)

    elif extension == ".xlsx":

        return extract_excel(path)

    elif extension == ".csv":

        return extract_csv(path)

    elif extension in [".png", ".jpg", ".jpeg"]:

        return extract_image(path)

    elif extension == ".txt":

        return extract_txt(path)

    else:

        raise ValueError(
            f"Unsupported file type : {extension}"
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logging.info("Industrial Document Ingestion Started")

    folder = r"D:\hackathons\data\nasa\archive\CMaps"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if os.path.isfile(path):

            print("\n" + "=" * 70)

            print(f"Reading : {file}")

            print("=" * 70)

            text = ingest_document(path)

            print(text[:500])

    logging.info("Document Ingestion Completed")