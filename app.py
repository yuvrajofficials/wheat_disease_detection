from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()
model = load_model('disease_detector.h5')
IMG_SIZE = (128, 128)

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    class_names = ['Healthy', 'Diseased']
    predicted_class = class_names[np.argmax(prediction[0])]

    return {"prediction": predicted_class, "confidence": float(np.max(prediction[0]))}
