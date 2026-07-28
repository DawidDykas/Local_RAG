from enum import Enum


class EventType(str, Enum):
    """
    Supported MinIO webhook events.
    """

    OBJECT_CREATED = "s3:ObjectCreated:Put"
    OBJECT_REMOVED_DELETE = "s3:ObjectRemoved:Delete"