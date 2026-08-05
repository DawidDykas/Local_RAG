import uuid
from datetime import datetime
from typing import Any
from urllib.parse import unquote_plus

from core.logger_config import logger
from infrastructure.vector_db.client import QdrantClientInstance
from qdrant_client.http import models
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

# =========================
# CONFIGURATION
# =========================

COLLECTION_NAME = "documents"
VECTOR_SIZE = 768  # for nomic-embed-text
DISTANCE = Distance.COSINE


class QdrantManager:
    """Manager class for Qdrant operations (save, search, delete)."""

    def __init__(self):
        """
        Initialize connection to Qdrant.

        Args:
            host: Hostname or IP address.
            port: Port (default 6333 for REST API).
        """
        self.client = QdrantClientInstance
        self.collection_name = COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        """Check if collection exists; if not, create it."""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
            )
            logger.debug(f"✅ Collection '{self.collection_name}' created!")
        else:
            logger.debug(f"ℹ️ Collection '{self.collection_name}' already exists")

    def save_embeddings(
        self,
        nodes: list[Any],
        embeddings: list[list[float]],
        file_name: str,
        bucket: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """
        Save embeddings to Qdrant.

        Args:
            nodes: List of LlamaIndex nodes (each has .text and .metadata).
            embeddings: List of embedding vectors.
            file_name: Name of the source file.
            bucket: Name of the MinIO bucket.
            metadata: Additional metadata to store.

        Returns:
            Dict with save statistics.
        """
        if len(nodes) != len(embeddings):
            raise ValueError("Number of nodes and embeddings must match!")

        points = []
        file_id = file_name.replace("/", "_").replace(".", "_").replace(" ", "_")

        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            point_id = f"{file_id}_chunk_{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

            # Prepare payload
            payload = {
                "text": node.text,
                "file_name": file_name,
                "bucket": bucket,
                "chunk_index": i,
                "total_chunks": len(nodes),
                "created_at": datetime.now().isoformat(),
            }

            # Add node metadata if present
            if hasattr(node, "metadata") and node.metadata:
                payload["metadata"] = node.metadata

            # Add additional metadata
            if metadata:
                payload.update(metadata)

            point = {
                "id": point_id,
                "vector": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                "payload": payload,
            }
            points.append(point)

        # Save to Qdrant
        try:
            logger.info(
                f"Saving {len(points)} points to Qdrant collection '{self.collection_name}'"
            )
            response = self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

            logger.debug(f"✅ Saved {len(points)} vectors to Qdrant")

            return {
                "status": "success",
                "points_count": len(points),
                "file_name": file_name,
                "collection": self.collection_name,
                "response": str(response),
            }

        except Exception as e:
            logger.error(f"❌ Error saving to Qdrant: {e}")
            raise

    def delete_embeddings_by_file_id(self, file_id: str) -> dict:
        """
        Delete all points associated with a given file.

        Args:
            file_id: File name or identifier.

        Returns:
            Dict with deletion statistics.
        """
        try:
            file_id = unquote_plus(file_id)

            # Search for points with this file_name
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=models.Query(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_name", match=models.MatchValue(value=file_id)
                            )
                        ]
                    )
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            point_ids = [point.id for point in search_result.points]

            if point_ids:
                # Delete points
                self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
                logger.debug(f"🗑️ Deleted {len(point_ids)} points for file: {file_id}")
            else:
                logger.debug(f" No points found for file: {file_id}")

            return {"status": "success", "deleted_count": len(point_ids), "file_name": file_id}

        except Exception as e:
            logger.error(f"❌ Error deleting: {e}")
            raise

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict | None = None,
        with_payload: bool = True,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score (0-1).
            filters: Dictionary of filters, e.g. {"file_name": "document.pdf"}.
            with_payload: Whether to return payload.

        Returns:
            List of results with payload.
        """
        # Prepare filter if provided
        filter_obj = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            filter_obj = Filter(must=conditions)

        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
            query_filter=filter_obj,
        )

        # Convert results to dicts
        output = []
        for result in results:
            output.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload if with_payload else None,
                    "vector": result.vector if hasattr(result, "vector") else None,
                }
            )

        logger.debug(f"🔍 Found {len(output)} results")
        return output

    def search_by_text(
        self,
        query_text: str,
        embed_function,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Search by text (automatically generates embedding).

        Args:
            query_text: Text query.
            embed_function: Function to generate embedding from text.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            filters: Dictionary of filters.

        Returns:
            List of results.
        """
        # Generate embedding for the query
        query_embedding = embed_function(query_text)

        # Search using the embedding
        return self.search(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
        )

    def delete_by_file(self, file_name: str) -> dict:
        """
        Delete all points associated with a given file name.

        Args:
            file_name: Name of the file.

        Returns:
            Dict with deletion statistics.
        """
        try:
            filter_obj = Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
            )

            # Retrieve all point IDs for the file
            points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            point_ids = [point.id for point in points[0]]

            if point_ids:
                # Delete points
                self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
                logger.debug(f"🗑️ Deleted {len(point_ids)} points for file: {file_name}")
            else:
                logger.debug(f"ℹ️ No points to delete for file: {file_name}")

            return {"status": "success", "deleted_count": len(point_ids), "file_name": file_name}

        except Exception as e:
            logger.error(f"❌ Error deleting: {e}")
            raise

    def get_collection_info(self) -> dict:
        """
        Retrieve information about the collection.

        Returns:
            Dict with collection stats.
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": info.config,
        }

    def delete_collection(self):
        """
        Delete the entire collection (IRREVERSIBLE!).
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.warning(f"🗑️ Collection '{self.collection_name}' has been deleted!")
        except Exception as e:
            logger.error(f"❌ Error deleting collection: {e}")
            raise


# =========================
# SINGLETON INSTANCE
# =========================

# Create a global instance (to be used across the application)
# qdrant_manager = QdrantManager()
