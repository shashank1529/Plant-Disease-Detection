from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
origins = [
    "http://localhost:3000",  # Allow requests from this origin
    # Add more origins if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

MODEL = tf.keras.models.load_model("models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello! I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(image_batch)
    idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[idx]
    confidence = np.max(prediction[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)