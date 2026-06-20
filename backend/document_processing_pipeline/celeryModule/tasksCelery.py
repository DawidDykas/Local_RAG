from .setting_celery import celery_broker
from enum import Enum
from src.Load_module.document_loader import process_file
from log_config.logger_config import logger
from qdrantBase.clientQdrant import qdrant_manager
from urllib.parse import unquote_plus

class EventType(str, Enum):
    OBJECT_CREATED = "s3:ObjectCreated:Put"
    OBJECT_REMOVED_DELETE = "s3:ObjectRemoved:Delete"



def handle_object_created(event_data: dict) -> None:
    logger.info(f"Handling object created event: {event_data}")
    resultat = process_file(
        bucket=event_data["Records"][0]["s3"]["bucket"]["name"],
        key=event_data["Records"][0]["s3"]["object"]["key"]
    )
    logger.info(resultat)
    logger.info(f"resultat keys: {resultat.keys()}")



def handle_object_removed(event_data: dict) -> None:
    file_key = event_data["Records"][0]["s3"]["object"]["key"]
    file_key = unquote_plus(file_key)  # decode URL
    
    logger.info(f"🗑️ Deleting embeddings for: {file_key}")
    
    resultat = qdrant_manager.delete_by_file(file_key)
    
    logger.info(f"✅ Deleted embeddings for: {file_key}, deleted: {resultat['deleted_count']} points")

methods = { 
    EventType.OBJECT_REMOVED_DELETE: handle_object_removed,
    EventType.OBJECT_CREATED: handle_object_created
}





@celery_broker.task
def webhook_minio_events(data: dict) -> None: 
    try:
        event_type = data.get("EventName")
        logger.debug(f"Event type: {event_type}")
        handler = methods.get(EventType(event_type))
        if handler:
            logger.info(f"Found handler for event type: {event_type}")
            handler(data)
        else:
            logger.warning(f"No handler found for event type: {event_type}")
    except Exception as e:
        logger.error(f"Error processing MinIO event: {e}", exc_info=True)