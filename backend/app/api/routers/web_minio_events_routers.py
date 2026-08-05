from fastapi import APIRouter, Request
from core.logger_config import logger
from workers.tasksCelery import webhook_minio_events

router = APIRouter()


@router.post(
    "/minio-event",
    summary="Receive MinIO webhook events",
    description=(
        "This endpoint receives webhook events from MinIO after file uploads. "
        "The received event data is logged and then passed to an asynchronous "
        "Celery task for further processing (e.g., file download, processing, indexing)."
    ),
    response_description="Returns the received event JSON or an error message",
)
async def handle_minio_event(req: Request) -> None:
    """
    Handle MinIO events - receives event data and logs it

    ### Process:
    1. Receives event data as JSON from the MinIO request
    2. Logs the received event
    3. Calls Celery task `webhook_minio_events.delay()` for asynchronous processing
    4. Returns the received data as acknowledgment

    ### Possible response codes:
    - **200**: Successfully received and forwarded to Celery
    - **400**: JSON parsing error or other client errors (handled by FastAPI)
    - **500**: Error while calling the Celery task
    """
    try:
        event_data = await req.json()
        logger.debug(f"Received MinIO event: {event_data}")

        try:
            logger.debug("Calling Celery task to handle MinIO event...")
            webhook_minio_events.delay(event_data)
        except Exception as e:
            logger.debug(f"❌ Error calling Celery task: {e}")
            return {"error": str(e)}

        return event_data
    except Exception as e:
        logger.error(f"❌ Error handling event: {e}")
        return {"error": str(e)}
