import io
from PIL import Image

import torch
import torchvision.transforms as T
from torchsummary import summary


from app.config import CONFIG
from model.model import build_efficientnet

CLASS_NAMES = ["fit", "overweight"]

def get_model():
    model = build_efficientnet()
    model.load_state_dict(torch.load(
        CONFIG["MODEL_PATH"], map_location=torch.device(CONFIG["DEVICE"])
    ))
    model.eval()
    return model


def transform_image(raw_image):
    transforms = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(raw_image))
    return transforms(image).unsqueeze(0)


def get_classification(model, raw_image):
    image_tensor = transform_image(raw_image)
    print("Image tensor:", image_tensor.shape)

    output = model(image_tensor)
    print("Output: ", output)
    predicted_class = torch.max(output, 1).indices.item()
    print("Predicted class:", predicted_class)
    return CLASS_NAMES[predicted_class]