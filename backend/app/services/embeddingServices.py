from infrastructure.minio.client import s3
from loaders.Load_module import PDFLoader, DOCXLoader, CSVLoader, HTMLLoader, TextLoader
from typing import List, Dict
from urllib.parse import unquote_plus

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from core.logger_config import logger

from services.qdrantServices import QdrantManager
from datetime import datetime
import re


class DocumentProcessor:
    """
    Service class for processing documents and storing embeddings.

    This class encapsulates the workflow of downloading a document
    from S3/MinIO, extracting text, chunking it, generating embeddings,
    and saving them to Qdrant vector database.
    """

    def __init__(self, qdrant_manager = QdrantManager(),
                 embed_model = OllamaEmbedding(
                                            model_name="nomic-embed-text",
                                            base_url="http://ollama:11434"
                                            ),
                 splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)):

        
        self.s3_client = s3
        self.embed_model = embed_model 
        self.splitter = splitter

        self.qdrant_manager = qdrant_manager


    # ROUTER
    # =========================
    @staticmethod
    def get_loader(ext: str) -> object:
        """
        Return document loader based on file extension.

        Args:
            ext (str):
                File extension without dot.

        Returns:
            object:
                Loader instance responsible for parsing document.

        Example:
            >>> get_loader("pdf")
            PDFLoader()

        """
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
    @staticmethod
    def chunk_text(text: str, max_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks.

        Text is divided by sentence boundaries and limited
        by maximum chunk size.

        Args:
            text (str):
                Input document text.

            max_size (int):
                Maximum number of characters per chunk.

        Returns:
            List[str]:
                List of text chunks.

        Example:
            >>> chunk_text("Hello world. This is test.")
            [
                "Hello world.",
                "This is test."
            ]
        """
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


    def process_file(self, bucket: str, key: str, s3 = s3) -> Dict:
        """
        Process document from object storage and store vector embeddings.

        This pipeline performs:
        
        1. Download file from MinIO/S3 bucket.
        2. Detect file extension.
        3. Select appropriate document loader.
        4. Extract raw text content.
        5. Split document into smaller chunks.
        6. Generate embeddings using Ollama embedding model.
        7. Save vectors and metadata into Qdrant vector database.

        Supported file formats:
            - PDF
            - DOCX
            - CSV
            - HTML
            - TXT

        Args:
            bucket (str):
                Name of MinIO/S3 bucket containing the file.

            key (str):
                Object key/path of the uploaded file.

            s3:
                S3 compatible client used to download the object.

        Returns:
            Dict:
                Result returned from Qdrant storage operation.
                Contains information about saved embeddings.

        Raises:
            Exception:
                If downloading, parsing, embedding generation,
                or vector storage fails.

        Example:
            >>> result = process_file(
            ...     bucket="documents",
            ...     key="reports/example.pdf"
            ... )

            {
                "status": "success",
                "vectors_saved": 25
            }

        """

        logger.debug(f"Processing file from bucket: {bucket}, key: {key}")
        key = unquote_plus(key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()

        ext = key.split(".")[-1].lower()

        loader = self.get_loader(ext)
        text = loader.load(data)

        doc = Document(text=text)
        nodes = self.splitter.get_nodes_from_documents([doc])
        logger.debug(f"📄 Number of chunks: {len(nodes)}")

        texts = [node.text for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(texts)

        result = self.qdrant_manager.save_embeddings(
            nodes=nodes,
            embeddings=embeddings,
            file_name=key,
            bucket=bucket,
            metadata={
                "file_extension": ext,
                "processed_at": str(datetime.now())
            }
        )
        
        logger.debug(f"✅ Saved to Qdrant: {result}")
        
        return result

