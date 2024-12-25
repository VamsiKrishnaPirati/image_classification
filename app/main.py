from fastapi import FastAPI, File, UploadFile
from app.routes import router

app = FastAPI()
app.include_router(router)
