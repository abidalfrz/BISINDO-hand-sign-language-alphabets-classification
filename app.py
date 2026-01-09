from fastapi import FastAPI, File, UploadFile, HTTPException, status
import cv2
import numpy as np
import uvicorn
from pydantic import BaseModel
from services.preprocessor import preprocess_images 
from services.predictor import predictor 

class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float

app = FastAPI(title="BISINDO Sign Language API")

@app.get("/")
def index():
    return {"message": "Welcome to the BISINDO Sign Language Prediction API!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file type. Please upload an image."
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
             raise HTTPException(status_code=400, detail="Could not decode image.")
        
        img_transform = preprocess_images(img, target_size=(224, 224))
        
        predicted_label, confidence = predictor.predict(img_transform)
        
        return {
            "predicted_label": predicted_label, 
            "confidence": float(confidence)
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    FILE_NAME = "app"
    ENTRY_POINT = "app"
    HOST = "127.0.0.1"
    PORT = 8000
    uvicorn.run(f"{FILE_NAME}:{ENTRY_POINT}", host=HOST, port=PORT, reload=True)

