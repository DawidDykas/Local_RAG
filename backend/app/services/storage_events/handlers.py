from urllib.parse import unquote_plus
from loaders.Load_module.document_loader import process_file
from services.QdrantServices import qdrant_manager
from core.logger_config import logger


def handle_object_created(event_data: dict) -> None:
    """
    Process newly uploaded MinIO object.
    """

    bucket = event_data["Records"][0]["s3"]["bucket"]["name"]
    key = event_data["Records"][0]["s3"]["object"]["key"]

    result = process_file(
        bucket=bucket,
        key=key
    )

    logger.debug(result)



def handle_object_removed(event_data: dict) -> None:
    """
    Remove document embeddings after object deletion.
    """

    key = event_data["Records"][0]["s3"]["object"]["key"]

    key = unquote_plus(key)

    result = qdrant_manager.delete_by_file(key)

    logger.debug(
        f"Deleted {result['deleted_count']} vectors"
    )