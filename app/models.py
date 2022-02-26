from typing import Optional, List
import io
from PIL import Image

from pydantic import BaseModel
from fastapi import UploadFile
import torch
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet

from app.config import CONFIG
from model.model import build_efficientnet


class PredictionInput(BaseModel):
    raw_image: UploadFile


class PredictionOutput(BaseModel):
    category: str


class EfficientNetClassifier:
    # model: Optional[EfficientNet]
    # targets: Optional[List[str]]

    def __init__(self, targets: List[str]) -> None:

        self.model: Optional[EfficientNet] = None
        self.targets = targets

    @classmethod
    def transform_image(cls, raw_image):
        transforms = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(raw_image))
        return transforms(image).unsqueeze(0)

    def load_model(self):
        self.model = build_efficientnet()
        self.model.load_state_dict(torch.load(
            CONFIG["MODEL_PATH"], map_location=torch.device(CONFIG["DEVICE"])
        ))
        self.model.eval()

    def get_classification(self, raw_image: UploadFile) -> PredictionOutput:
        image_tensor = self.transform_image(raw_image.file.read())

        output = self.model(image_tensor)
        print("Output: ", output)
        predicted_class = torch.max(output, 1).indices.item()
        print("Predicted class:", predicted_class)
        category = self.targets[predicted_class]
        return PredictionOutput(category=category)

# TODO: Implement PredictionInput and Outout model
