from fastapi import FastAPI
from api.routers import web_minio_events_routers
from api.routers import model_routers

app = FastAPI()

app.include_router(
    model_routers.router_modelOLLAMA,
    tags=["Model OLLAMA"]
)

app.include_router(
    web_minio_events_routers.router,
    tags=["MinIO Events"]
)


