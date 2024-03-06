from fastapi import FastAPI
from v1.routes.base import router as v1_router

app = FastAPI()
app.include_router(v1_router, prefix="/v1", tags=["v1_router"])
