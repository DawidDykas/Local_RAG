from .document_loader import process_file
from .loader import CSVLoader, DOCXLoader, HTMLLoader, PDFLoader, TextLoader

__all__ = ["CSVLoader", "DOCXLoader", "HTMLLoader", "PDFLoader", "TextLoader", "process_file"]
