from backend.minio_server.server_client import s3
from .loader import PDFLoader, DOCXLoader, CSVLoader, HTMLLoader, TextLoader


# =========================
# ROUTER
# =========================

def get_loader(ext: str):
    loaders = {
        "pdf": PDFLoader(),
        "docx": DOCXLoader(),
        "csv": CSVLoader(),
        "html": HTMLLoader(),
        "txt": TextLoader(),
    }

    return loaders.get(ext, TextLoader())


# =========================
# CHUNKING
# =========================
import re
def chunk_text(text: str, max_size: int = 500):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        # jeśli pojedyncze zdanie jest bardzo długie
        if len(sentence) > max_size:
            if current:
                chunks.append(current)
                current = ""

            # fallback: tniemy długie zdanie
            for i in range(0, len(sentence), max_size):
                chunks.append(sentence[i:i+max_size])

            continue

        # normalne dodawanie
        if len(current) + len(sentence) + 1 <= max_size:
            current += (" " + sentence if current else sentence)
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks

# =========================
# MAIN PIPELINE
# =========================

def process_file(bucket: str, key: str, s3 = s3) -> dict:
    # 1. pobierz z MinIO
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    # 2. wykryj typ
    ext = key.split(".")[-1].lower()

    # 3. wybierz loader
    loader = get_loader(ext)

    # 4. parsuj do tekstu
    text = loader.load(data)

    # 5. chunking
    chunks = chunk_text(text)

    return {
        "file": key,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "key": key,
        "bucket": bucket
    }