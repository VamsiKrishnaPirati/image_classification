from fastapi import APIRouter, File, UploadFile
from model.predict import predict_image
import os

router = APIRouter()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_location = f"data/sample_images/{file.filename}"
    os.makedirs("data/sample_images", exist_ok=True)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    prediction = predict_image(file_location)
    return {"filename": file.filename, "prediction": prediction}
