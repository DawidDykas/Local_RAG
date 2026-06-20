from ..server_client import s3
from .loader import PDFLoader, DOCXLoader, CSVLoader, HTMLLoader, TextLoader
from typing import List, Dict
from urllib.parse import unquote_plus

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from log_config.logger_config import logger

from qdrantBase.clientQdrant import qdrant_manager
from datetime import datetime


# embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://ollama:11434"
)

# chunking (jawny żeby mieć kontrolę)
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)



# =========================
# ROUTER
# =========================

def get_loader(ext: str) -> object:
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
def chunk_text(text: str, max_size: int = 500) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)


    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_size):
                chunks.append(sentence[i:i+max_size])

            continue

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
# 
# def process_file(bucket: str, key: str, s3 = s3) -> List[Dict[str, any]]:
#     # 1. pobierz z MinIO
#     logger.info(f"Processing file from bucket: {bucket}, key: {key}")

#     key = unquote_plus(key)

#     obj = s3.get_object(Bucket=bucket, Key=key)
#     data = obj["Body"].read()

#     # 2. wykryj typ
#     ext = key.split(".")[-1].lower()

#     # 3. wybierz loader
#     loader = get_loader(ext)

#     # 4. parsuj do tekstu
#     text = loader.load(data)

#     # 5. chunking
#     # chunks = chunk_text(text)


#     doc = Document(text=text)

#     nodes = splitter.get_nodes_from_documents([doc])
#     logger.info(len(nodes))



#     texts = [node.text for node in nodes]
#     embeddings = embed_model.get_text_embedding_batch(texts)

#     results = []
#     for node, emb in zip(nodes, embeddings):
#         results.append({
#             "text": node.text,
#             "metadata": node.metadata,
#             "embedding": emb
#         })


def process_file(bucket: str, key: str, s3 = s3) -> Dict:
    # 1. pobierz z MinIO
    logger.info(f"Processing file from bucket: {bucket}, key: {key}")
    key = unquote_plus(key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    # 2. wykryj typ
    ext = key.split(".")[-1].lower()

    # 3. wybierz loader
    loader = get_loader(ext)

    # 4. parsuj do tekstu
    text = loader.load(data)

    # 5. chunking
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    logger.info(f"📄 Liczba chunków: {len(nodes)}")

    # 6. generuj embeddingi
    texts = [node.text for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(texts)

    result = qdrant_manager.save_embeddings(
        nodes=nodes,
        embeddings=embeddings,
        file_name=key,
        bucket=bucket,
        metadata={
            "file_extension": ext,
            "processed_at": str(datetime.now())
        }
    )
    
    logger.info(f"✅ Zapisano do Qdrant: {result}")
    
    return result

