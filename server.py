import os
import cv2
import torch
import torchvision
from PIL import Image
import time
import numpy as np
from pydantic import BaseModel
import pandas as pd
import io
import base64
import json
import uvicorn
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

class StreamInput(BaseModel):
    b64str: str
    model: str

model_dice = torch.hub.load(
    "ultralytics/yolov5", 
    "custom", path="weights/best.pt", 
    force_reload=False, 
    device="cuda" if torch.cuda.is_available() else "cpu")
model_dice.conf = 0.6
model_dice.iou = 0.01
model_test = torch.hub.load("ultralytics/yolov5", "yolov5s")

app = FastAPI(title="Dice detection")

@app.get("/")
def home():
    return "Server is running. Go to http://localhost:8080/docs"

@app.post("/file/to-img")
def detect_return_image(image: UploadFile):
    
    image_stream = io.BytesIO(image.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = model_test(image)
    result.render()
    for img in result.imgs:
        detected = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    _, encoded = cv2.imencode(".jpg", detected)
    byte_output = encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(byte_output), media_type="image/jpg")
    
@app.post("/file/to-b64")
def detect_return_base64(image: UploadFile):
    
    image_stream = io.BytesIO(image.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = model_test(image)
    result.render()
    for img in result.imgs:
        buffered = io.BytesIO()
        img_object = Image.fromarray(img)
        img_object.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue())
    
    return img_base64


@app.post("/file/to-json")
def detect_return_labels(image: UploadFile):
    
    image_stream = io.BytesIO(image.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model_test(image)
    labels = result.pandas().xyxy[0].to_json(orient="records")
    
    labels_json = json.loads(labels)
    
    return labels_json

@app.post("/predict/to-img")
def detect_return_image(client_input: StreamInput):
    image_bytes = base64.b64decode(client_input.b64str.encode("utf-8"))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8) 
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if client_input.model == "Dice":
        result = model_dice(image)
    else:
        result = model_test(image)
    result.render()
    for img in result.imgs:
        detected = img#cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", detected)
    byte_output = encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(byte_output), media_type="image/jpg")
    
@app.post("/predict/to-b64")
def detect_return_base64(client_input: StreamInput):
    image_bytes = base64.b64decode(client_input.b64str.encode("utf-8"))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8) 
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if client_input.model == "Dice":
        result = model_dice(image)
    else:
        result = model_test(image)
    result.render()
    for img in result.imgs:
        buffered = io.BytesIO()
        img_object = Image.fromarray(img)
        img_object.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue())
    
    return img_base64


@app.post("/predict/to-json")
def detect_return_labels(client_input: StreamInput):
    image_bytes = base64.b64decode(client_input.b64str.encode("utf-8"))
    image_array = np.frombuffer(image_bytes, dtype=np.uint8) 
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if client_input.model == "Dice":
        result = model_dice(image)
    else:
        result = model_test(image)
    
    df = result.pandas().xyxy[0]
    labels = df.to_json(orient="records")
    labels_json = json.loads(labels)

    return labels_json

nest_asyncio.apply()

host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

uvicorn.run(app, host=host, port=8080)