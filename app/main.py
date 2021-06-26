"""
python module for creating REST APIs
using FastAPI framework
"""

import os
from tensorflow.python.distribute.multi_worker_util import worker_count
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = FastAPI()

origins = ["https://zipnet.herokuapp.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("../model/zipnet")


@app.get("/")
async def root():
    return {"message": "This is Crowd Counting API based on ZIPNet!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Args:
        file (Images): image file
    Returns:
        prediction: probability of environment without people and predicted human counts
    """
    img = tf.image.decode_jpeg(await file.read(), channels=3)
    img = tf.image.resize(img, [224, 224]) / 255
    lam, p = model.predict(img[None, :, :, :])
    lam = lam.flatten()
    p = p.flatten()
    pred = (1 - p) * lam

    return {"message": "ok", "p": np.asscalar(p), "pred": np.asscalar(pred)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", default=8000)),
        reload=True,
        workers=1,
    )
