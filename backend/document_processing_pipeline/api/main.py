from fastapi import FastAPI
from api.routers import web_minio_events_router
import uvicorn

app = FastAPI()



app.include_router(
    web_minio_events_router.router,
    tags=["MinIO Events"]
)


