import uuid
from unittest.mock import Mock, patch

import pytest
from qdrant_client.models import Distance, VectorParams
from services.qdrantServices import COLLECTION_NAME, DISTANCE, VECTOR_SIZE, QdrantManager

# =========================
# FIXTURES
# =========================


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for unit testing."""
    with patch("services.QdrantServices.QdrantClientInstance") as mock_client:
        # Mock collection methods
        mock_client.get_collections.return_value = Mock(collections=[Mock(name="other_collection")])
        mock_client.create_collection.return_value = None
        mock_client.upsert.return_value = Mock(status="completed")
        mock_client.query_points.return_value = Mock(points=[])
        mock_client.delete.return_value = None
        mock_client.search.return_value = []
        mock_client.scroll.return_value = ([], None)
        mock_client.get_collection.return_value = Mock(
            vectors_count=100,
            points_count=100,
            status="green",
            config={"params": {"vectors": {"size": 768}}},
        )
        mock_client.delete_collection.return_value = None

        yield mock_client


@pytest.fixture
def qdrant_manager(mock_qdrant_client):
    """Create QdrantManager instance with mocked client."""
    return QdrantManager()


@pytest.fixture
def sample_nodes():
    """Create sample nodes for testing."""

    class MockNode:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}
            self.node_id = str(uuid.uuid4())

    return [
        MockNode("This is the first chunk of text.", {"page": 1, "section": "intro"}),
        MockNode("This is the second chunk of text.", {"page": 2, "section": "body"}),
        MockNode("This is the third chunk of text.", {"page": 3, "section": "conclusion"}),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3] * 256,  # 768 dimensions
        [0.4, 0.5, 0.6] * 256,
        [0.7, 0.8, 0.9] * 256,
    ]


@pytest.fixture
def mock_point():
    """Create a mock point for search results."""
    point = Mock()
    point.id = "test_point_id"
    point.score = 0.95
    point.payload = {
        "text": "Sample text content",
        "file_name": "test.pdf",
        "bucket": "test_bucket",
        "chunk_index": 0,
        "created_at": "2024-01-15T10:00:00",
    }
    point.vector = [0.1, 0.2, 0.3] * 256
    return point


# =========================
# INITIALIZATION TESTS
# =========================


class TestQdrantManagerInitialization:
    """Unit tests for QdrantManager initialization."""

    def test_init_collection_exists(self, mock_qdrant_client):
        """Test initialization when collection already exists."""
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name=COLLECTION_NAME)]
        )

        manager = QdrantManager()

        mock_qdrant_client.create_collection.assert_not_called()
        assert manager.collection_name == COLLECTION_NAME

    def test_init_collection_not_exists(self, mock_qdrant_client):
        """Test initialization when collection does not exist."""
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        manager = QdrantManager()

        mock_qdrant_client.create_collection.assert_called_once_with(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )

    def test_init_collection_empty(self, mock_qdrant_client):
        """Test initialization when no collections exist."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        manager = QdrantManager()

        mock_qdrant_client.create_collection.assert_called_once()

    def test_init_collection_error(self, mock_qdrant_client):
        """Test initialization when collection creation fails."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")

        with pytest.raises(Exception, match="Creation failed"):
            manager = QdrantManager()


# =========================
# SAVE EMBEDDINGS TESTS
# =========================


class TestSaveEmbeddings:
    """Unit tests for save_embeddings method."""

    def test_save_embeddings_success(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test successful saving of embeddings."""
        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name="test_document.pdf",
            bucket="test_bucket",
            metadata={"author": "Test Author", "version": "1.0"},
        )

        # Verify result
        assert result["status"] == "success"
        assert result["points_count"] == 3
        assert result["file_name"] == "test_document.pdf"
        assert result["collection"] == COLLECTION_NAME
        assert result["response"] is not None

        # Verify upsert called
        qdrant_manager.client.upsert.assert_called_once()

        # Verify point structure
        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert len(points) == 3
        assert call_args.kwargs["wait"] is True

        # Check first point
        point = points[0]
        assert "id" in point
        assert "vector" in point
        assert "payload" in point
        assert point["payload"]["text"] == "This is the first chunk of text."
        assert point["payload"]["file_name"] == "test_document.pdf"
        assert point["payload"]["bucket"] == "test_bucket"
        assert point["payload"]["chunk_index"] == 0
        assert point["payload"]["total_chunks"] == 3
        assert point["payload"]["author"] == "Test Author"
        assert point["payload"]["version"] == "1.0"
        assert "created_at" in point["payload"]

    def test_save_embeddings_nodes_metadata(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test saving embeddings with node metadata."""
        # Nodes already have metadata from fixture
        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name="test.pdf",
            bucket="test_bucket",
        )

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]

        # Check first point has metadata
        assert "metadata" in points[0]["payload"]
        assert points[0]["payload"]["metadata"]["page"] == 1
        assert points[0]["payload"]["metadata"]["section"] == "intro"

    def test_save_embeddings_mismatch_error(self, qdrant_manager, sample_nodes):
        """Test error when nodes and embeddings count mismatch."""
        embeddings = [[0.1, 0.2, 0.3] * 256]  # Only 1 embedding

        with pytest.raises(ValueError, match="Number of nodes and embeddings must match!"):
            qdrant_manager.save_embeddings(
                nodes=sample_nodes,
                embeddings=embeddings,
                file_name="test.pdf",
                bucket="test_bucket",
            )

    def test_save_embeddings_empty_data(self, qdrant_manager):
        """Test error with empty nodes and embeddings."""
        with pytest.raises(ValueError, match="Number of nodes and embeddings must match!"):
            qdrant_manager.save_embeddings(
                nodes=[], embeddings=[], file_name="empty.pdf", bucket="test_bucket"
            )

    def test_save_embeddings_single_point(self, qdrant_manager):
        """Test saving a single embedding."""

        class MockNode:
            def __init__(self):
                self.text = "Single chunk"
                self.metadata = {}
                self.node_id = str(uuid.uuid4())

        node = MockNode()
        embedding = [0.5] * 768

        result = qdrant_manager.save_embeddings(
            nodes=[node], embeddings=[embedding], file_name="single.pdf", bucket="test_bucket"
        )

        assert result["points_count"] == 1

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0]["payload"]["total_chunks"] == 1
        assert points[0]["payload"]["chunk_index"] == 0

    def test_save_embeddings_without_metadata(
        self, qdrant_manager, sample_nodes, sample_embeddings
    ):
        """Test saving embeddings without additional metadata."""
        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name="test.pdf",
            bucket="test_bucket",
        )

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]

        # Should have basic fields but not extra metadata
        assert "author" not in points[0]["payload"]
        assert "version" not in points[0]["payload"]
        assert "text" in points[0]["payload"]
        assert "file_name" in points[0]["payload"]
        assert "bucket" in points[0]["payload"]

    def test_save_embeddings_uuid_generation(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test that unique UUIDs are generated for each point."""
        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name="test.pdf",
            bucket="test_bucket",
        )

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]

        # All IDs should be unique
        ids = [p["id"] for p in points]
        assert len(set(ids)) == len(ids)

        # IDs should be valid UUID format (UUID v5)
        for point_id in ids:
            try:
                uuid.UUID(point_id)
            except ValueError:
                pytest.fail(f"Invalid UUID: {point_id}")

    def test_save_embeddings_error_handling(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test error handling when Qdrant upsert fails."""
        qdrant_manager.client.upsert.side_effect = Exception("Network timeout")

        with pytest.raises(Exception, match="Network timeout"):
            qdrant_manager.save_embeddings(
                nodes=sample_nodes,
                embeddings=sample_embeddings,
                file_name="test.pdf",
                bucket="test_bucket",
            )


# =========================
# DELETE TESTS
# =========================


class TestDelete:
    """Unit tests for delete methods."""

    def test_delete_embeddings_by_file_id_success(self, qdrant_manager):
        """Test successful deletion by file ID with points found."""
        mock_point = Mock()
        mock_point.id = "point_1"
        qdrant_manager.client.query_points.return_value = Mock(points=[mock_point])

        result = qdrant_manager.delete_embeddings_by_file_id("test_document.pdf")

        assert result["status"] == "success"
        assert result["deleted_count"] == 1
        assert result["file_name"] == "test_document.pdf"

        # Verify query was called
        qdrant_manager.client.query_points.assert_called_once()
        call_args = qdrant_manager.client.query_points.call_args
        assert call_args.kwargs["collection_name"] == COLLECTION_NAME
        assert call_args.kwargs["limit"] == 10000

        # Verify delete was called
        qdrant_manager.client.delete.assert_called_once()
        delete_args = qdrant_manager.client.delete.call_args
        assert delete_args.kwargs["collection_name"] == COLLECTION_NAME
        assert delete_args.kwargs["points_selector"] == ["point_1"]

    def test_delete_embeddings_by_file_id_no_points(self, qdrant_manager):
        """Test deletion when no points found."""
        qdrant_manager.client.query_points.return_value = Mock(points=[])

        result = qdrant_manager.delete_embeddings_by_file_id("nonexistent.pdf")

        assert result["status"] == "success"
        assert result["deleted_count"] == 0

        # Verify delete was not called
        qdrant_manager.client.delete.assert_not_called()

    def test_delete_embeddings_by_file_id_url_encoded(self, qdrant_manager):
        """Test deletion with URL-encoded file name."""
        mock_point = Mock()
        mock_point.id = "point_1"
        qdrant_manager.client.query_points.return_value = Mock(points=[mock_point])

        file_name = "test%20document.pdf"
        result = qdrant_manager.delete_embeddings_by_file_id(file_name)

        # Should decode the URL-encoded string
        call_args = qdrant_manager.client.query_points.call_args
        # The filter should have the decoded value
        filter_obj = call_args.kwargs["query"]
        assert filter_obj.filter.must[0].match.value == file_name

    def test_delete_by_file_success(self, qdrant_manager):
        """Test successful deletion by file name with points found."""
        mock_point = Mock()
        mock_point.id = "point_1"
        qdrant_manager.client.scroll.return_value = ([mock_point], None)

        result = qdrant_manager.delete_by_file("test_document.pdf")

        assert result["status"] == "success"
        assert result["deleted_count"] == 1

        # Verify scroll was called
        qdrant_manager.client.scroll.assert_called_once()
        call_args = qdrant_manager.client.scroll.call_args
        assert call_args.kwargs["collection_name"] == COLLECTION_NAME
        assert call_args.kwargs["limit"] == 10000

        # Verify delete was called
        qdrant_manager.client.delete.assert_called_once()

    def test_delete_by_file_no_points(self, qdrant_manager):
        """Test deletion by file name when no points found."""
        qdrant_manager.client.scroll.return_value = ([], None)

        result = qdrant_manager.delete_by_file("nonexistent.pdf")

        assert result["status"] == "success"
        assert result["deleted_count"] == 0
        qdrant_manager.client.delete.assert_not_called()

    def test_delete_error_handling(self, qdrant_manager):
        """Test error handling during deletion."""
        qdrant_manager.client.query_points.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            qdrant_manager.delete_embeddings_by_file_id("test.pdf")


# =========================
# SEARCH TESTS
# =========================


class TestSearch:
    """Unit tests for search methods."""

    def test_search_success(self, qdrant_manager, mock_point):
        """Test successful search."""
        qdrant_manager.client.search.return_value = [mock_point]

        query_vector = [0.1, 0.2, 0.3] * 256
        results = qdrant_manager.search(
            query_vector=query_vector, limit=10, score_threshold=0.7, with_payload=True
        )

        assert len(results) == 1
        assert results[0]["id"] == "test_point_id"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["text"] == "Sample text content"
        assert results[0]["payload"]["file_name"] == "test.pdf"

        # Verify search called with correct parameters
        qdrant_manager.client.search.assert_called_once_with(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=10,
            score_threshold=0.7,
            with_payload=True,
            query_filter=None,
        )

    def test_search_with_filters(self, qdrant_manager, mock_point):
        """Test search with filters."""
        qdrant_manager.client.search.return_value = [mock_point]

        filters = {"file_name": "test.pdf", "bucket": "test_bucket"}
        query_vector = [0.1, 0.2, 0.3] * 256

        results = qdrant_manager.search(
            query_vector=query_vector, limit=5, filters=filters, with_payload=False
        )

        # Verify filter was created
        call_args = qdrant_manager.client.search.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 2

        # Verify filter conditions
        filter_keys = [cond.key for cond in query_filter.must]
        assert "file_name" in filter_keys
        assert "bucket" in filter_keys

        # Verify with_payload=False
        assert call_args.kwargs["with_payload"] is False

    def test_search_with_score_threshold(self, qdrant_manager, mock_point):
        """Test search with score threshold."""
        qdrant_manager.client.search.return_value = [mock_point]

        query_vector = [0.1, 0.2, 0.3] * 256
        results = qdrant_manager.search(
            query_vector=query_vector, limit=5, score_threshold=0.9, with_payload=True
        )

        # Verify score threshold was passed
        call_args = qdrant_manager.client.search.call_args
        assert call_args.kwargs["score_threshold"] == 0.9

    def test_search_no_results(self, qdrant_manager):
        """Test search with no results."""
        qdrant_manager.client.search.return_value = []

        query_vector = [0.1, 0.2, 0.3] * 256
        results = qdrant_manager.search(query_vector=query_vector, limit=5)

        assert len(results) == 0

    def test_search_by_text_success(self, qdrant_manager, mock_point):
        """Test search by text using embedding function."""

        def mock_embed(text):
            return [0.1, 0.2, 0.3] * 256

        qdrant_manager.client.search.return_value = [mock_point]

        results = qdrant_manager.search_by_text(
            query_text="test query",
            embed_function=mock_embed,
            limit=10,
            score_threshold=0.8,
            filters={"file_name": "test.pdf"},
        )

        assert len(results) == 1

        # Verify search was called with embedding from mock_embed
        call_args = qdrant_manager.client.search.call_args
        assert call_args.kwargs["query_vector"] == [0.1, 0.2, 0.3] * 256
        assert call_args.kwargs["limit"] == 10
        assert call_args.kwargs["score_threshold"] == 0.8

    def test_search_by_text_empty_query(self, qdrant_manager):
        """Test search by text with empty query."""

        def mock_embed(text):
            return []

        with pytest.raises(Exception):
            qdrant_manager.search_by_text(query_text="", embed_function=mock_embed, limit=5)


# =========================
# COLLECTION INFO TESTS
# =========================


class TestCollectionInfo:
    """Unit tests for collection information methods."""

    def test_get_collection_info(self, qdrant_manager):
        """Test retrieving collection information."""
        info = qdrant_manager.get_collection_info()

        assert info["name"] == COLLECTION_NAME
        assert info["vectors_count"] == 100
        assert info["points_count"] == 100
        assert info["status"] == "green"
        assert "config" in info

        qdrant_manager.client.get_collection.assert_called_once_with(
            collection_name=COLLECTION_NAME
        )

    def test_delete_collection(self, qdrant_manager):
        """Test deleting the entire collection."""
        qdrant_manager.delete_collection()

        qdrant_manager.client.delete_collection.assert_called_once_with(
            collection_name=COLLECTION_NAME
        )

    def test_delete_collection_error(self, qdrant_manager):
        """Test error when deleting collection fails."""
        qdrant_manager.client.delete_collection.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            qdrant_manager.delete_collection()


# =========================
# EDGE CASES AND ERROR HANDLING
# =========================


class TestEdgeCases:
    """Unit tests for edge cases and error scenarios."""

    def test_file_name_with_special_characters(
        self, qdrant_manager, sample_nodes, sample_embeddings
    ):
        """Test file names with special characters."""
        file_name = "test document with spaces & special chars!@#$.pdf"

        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name=file_name,
            bucket="test_bucket",
        )

        assert result["status"] == "success"

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0]["payload"]["file_name"] == file_name

    def test_file_name_unicode(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test file names with Unicode characters."""
        file_name = "测试文档.pdf"

        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes,
            embeddings=sample_embeddings,
            file_name=file_name,
            bucket="test_bucket",
        )

        assert result["status"] == "success"

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0]["payload"]["file_name"] == file_name

    def test_bucket_with_special_characters(self, qdrant_manager, sample_nodes, sample_embeddings):
        """Test bucket names with special characters."""
        bucket = "test-bucket_123"

        result = qdrant_manager.save_embeddings(
            nodes=sample_nodes, embeddings=sample_embeddings, file_name="test.pdf", bucket=bucket
        )

        assert result["status"] == "success"

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert points[0]["payload"]["bucket"] == bucket

    def test_large_embedding_vector(self, qdrant_manager):
        """Test saving with large embedding vectors."""

        class MockNode:
            def __init__(self):
                self.text = "Test text"
                self.metadata = {}
                self.node_id = str(uuid.uuid4())

        large_embedding = [0.5] * 768  # 768 dimensions

        result = qdrant_manager.save_embeddings(
            nodes=[MockNode()],
            embeddings=[large_embedding],
            file_name="large_vector.txt",
            bucket="test_bucket",
        )

        assert result["status"] == "success"
        assert result["points_count"] == 1

    def test_special_characters_in_text(self, qdrant_manager):
        """Test saving text with special characters."""

        class MockNode:
            def __init__(self):
                self.text = "Text with special chars: @#$%^&*()_+{}|:<>?~`"
                self.metadata = {}
                self.node_id = str(uuid.uuid4())

        embedding = [0.5] * 768

        result = qdrant_manager.save_embeddings(
            nodes=[MockNode()], embeddings=[embedding], file_name="test.txt", bucket="test_bucket"
        )

        assert result["status"] == "success"

        call_args = qdrant_manager.client.upsert.call_args
        points = call_args.kwargs["points"]
        assert "special" in points[0]["payload"]["text"]


# =========================
# CONFIGURATION TESTS
# =========================


class TestConfiguration:
    """Unit tests for configuration constants."""

    def test_collection_name_constant(self):
        """Test COLLECTION_NAME constant."""
        assert COLLECTION_NAME == "documents"

    def test_vector_size_constant(self):
        """Test VECTOR_SIZE constant."""
        assert VECTOR_SIZE == 768  # nomic-embed-text dimension

    def test_distance_constant(self):
        """Test DISTANCE constant."""
        assert DISTANCE == Distance.COSINE


# =========================
# CONCURRENCY TESTS
# =========================


class TestConcurrency:
    """Unit tests for concurrent operations."""

    def test_concurrent_saves(self, qdrant_manager):
        """Test multiple concurrent saves."""
        import threading

        results = []
        errors = []

        def save_worker(worker_id):
            try:

                class MockNode:
                    def __init__(self):
                        self.text = f"Chunk {worker_id}"
                        self.metadata = {}
                        self.node_id = str(uuid.uuid4())

                embedding = [0.5] * 768
                result = qdrant_manager.save_embeddings(
                    nodes=[MockNode()],
                    embeddings=[embedding],
                    file_name=f"concurrent_{worker_id}.txt",
                    bucket="test_bucket",
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=save_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)


# =========================
# RUN TESTS
# =========================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--strict-markers"])
