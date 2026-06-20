from fastapi import APIRouter, Request
from log_config.logger_config import logger
from celeryModule.tasksCelery import webhook_minio_events

router = APIRouter()
logger.info("🚀 MinIO Events Router Initialized")


@router.post("/minio-event")
async def handle_minio_event(req: Request):
    """Handle MinIO events - receives event data and logs it"""
    try:
        event_data = await req.json()
        logger.info(f"Received MinIO event: {event_data}")

        # Call the Celery task to handle the event
        try:
            logger.info("Calling Celery task to handle MinIO event...")
            webhook_minio_events.delay(event_data)
        except Exception as e:
            logger.info(f"❌ Error calling Celery task: {e}")
            return {"ok": False, "error": str(e)}

        return event_data
    except Exception as e:
        logger.error(f"❌ Error handling event: {e}")
        return {"ok": False, "error": str(e)}
    


