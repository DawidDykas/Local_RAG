import os
import boto3
from dotenv import load_dotenv

# load_dotenv("backend/.env")  # Load environment variables from .env file


s3 = boto3.client(
    "s3",
    endpoint_url=f"http://minio:9000",
    aws_access_key_id=f'minio',
    aws_secret_access_key=f'minio123',
)

