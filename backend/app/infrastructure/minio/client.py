import boto3
from core.global_config import minio_database_settings

s3 = boto3.client(
    "s3",
    endpoint_url=minio_database_settings.url_minio_database,
    aws_access_key_id=minio_database_settings.user_name_acces,
    aws_secret_access_key=minio_database_settings.user_password_acces,
)
