from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "my_tomato_leaf_model.keras")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

model = model = tf.keras.models.load_model(MODEL_PATH)#, compile=False)

class_names = ['Bacterial_spot',
 'Early_blight',
 'Late_blight',
 'Leaf_Mold',
 'Septoria_leaf_spot',
 'Spider_mites Two-spotted_spider_mite',
 'Target_Spot',
 'Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato_mosaic_virus',
 'healthy']


@app.get("/")
def health():
    return {"status": "API running"}

def read_img(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")  # force 3 channels
    image = image.resize((224, 224))                     # correct size
    image = np.array(image).astype("float32") / 255.0   # normalize
    return image


@app.post("/predict")
async def predict_image(img: UploadFile = File(...)):
    
    image = read_img(await img.read())
    # image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image), axis=0)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction[0])

    return {
        "predicted_class": class_names[idx],
        "confidence": round(float(np.max(prediction)), 4)
    }
