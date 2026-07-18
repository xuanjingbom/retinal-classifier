import sys
import types
import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

# Register custom ChannelAttention so best.pt loads correctly
class ChannelAttention(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

mod = types.ModuleType('ultralytics.nn.modules.channel_attention')
mod.ChannelAttention = ChannelAttention
sys.modules['ultralytics.nn.modules.channel_attention'] = mod
import ultralytics.nn.modules as _ult_mod
_ult_mod.ChannelAttention = ChannelAttention

app = FastAPI(title="Retinal Disease Classifier API")
model = YOLO("best.pt")

@app.get("/")
def root():
    return {"status": "ok", "message": "Retinal Disease Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image, imgsz=224, verbose=False)
    probs = results[0].probs
    names = results[0].names

    all_probs = {names[i]: round(float(probs.data[i]), 4) for i in range(len(names))}

    return {
        "predicted_class": names[probs.top1],
        "confidence": round(float(probs.top1conf), 4),
        "all_class_probabilities": all_probs,
    }