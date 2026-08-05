# from config.setting import celery_settings
from core.global_config import celery_settings

from celery import Celery

try:
    celery_app = Celery(
        "worker", broker_url=celery_settings.url_broker, result_backend=celery_settings.url_broker
    )

except Exception as e:
    ValueError(f"Error initializing Celery: {e}")
