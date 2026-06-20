# from config.setting import celery_settings  
from celery import Celery
from log_config.logger_config import logger

try: 
    celery_broker = Celery("worker", 
                            broker_url = "redis://redis:6379/0", 
                            result_backend = "redis://redis:6379/0")
except Exception as e:
    logger.error(f"Error initializing Celery: {e}")

