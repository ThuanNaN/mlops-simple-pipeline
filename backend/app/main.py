from fastapi import FastAPI
from middleware import CORSMiddleware, origins, LogProcessAndTime
from v1.routes.base import router as v1_router

app = FastAPI()

app.add_middleware(CORSMiddleware, 
                   allow_origins=origins, 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"])
app.add_middleware(LogProcessAndTime)

app.include_router(v1_router, prefix="/v1", tags=["v1_router"])

