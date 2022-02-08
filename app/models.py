from typing import Optional

from pydantic import BaseModel

# import io
# from PIL import Image

# from fastapi import UploadFile

# import torch
# import torchvision.transforms as T
# from torchsummary import summary


# from app.config import CONFIG
# from app.models import PredictionOutput
# from model.model import build_efficientnet

# CLASS_NAMES = ["fit", "overweight"]

# def get_model():
#     model = build_efficientnet()
#     model.load_state_dict(torch.load(
#         CONFIG["MODEL_PATH"], map_location=torch.device(CONFIG["DEVICE"])
#     ))
#     model.eval()
#     return model


# class EfficientNet:
#     model: Optional[]


# class PredictionInput(BaseModel):
#     raw_image: bytes


class PredictionOutput(BaseModel):
    category: str

