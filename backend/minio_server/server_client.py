import os
import boto3
from dotenv import load_dotenv

load_dotenv("backend/.env")  # Load environment variables from .env file
print(os.getenv('MINIO_SERVER_AD'))
s3 = boto3.client(
    "s3",
    endpoint_url=f"http://localhost:{os.getenv('MINIO_SERVER_AD')}",
    aws_access_key_id=os.getenv("USER_NAME_ACCES"),
    aws_secret_access_key=os.getenv("USER_PASSWORD_ACCES")
)