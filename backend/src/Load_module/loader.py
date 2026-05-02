import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
from enum import Enum
import io
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from bs4 import BeautifulSoup


# =========================
# LOADERS
# =========================

class PDFLoader:
    def load(self, data: bytes):
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)


class DOCXLoader:
    def load(self, data: bytes):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)


class CSVLoader:
    def load(self, data: bytes):
        df = pd.read_csv(io.BytesIO(data))
        return df.to_string(index=False)


class HTMLLoader:
    def load(self, data: bytes):
        soup = BeautifulSoup(data, "lxml")
        return soup.get_text()


class TextLoader:
    def load(self, data: bytes):
        return data.decode("utf-8", errors="ignore")