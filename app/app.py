from fastapi import FastAPI, UploadFile, Depends
from fastapi.logger import logger
from fastapi.exceptions import RequestValidationError

from app.config import CONFIG
from app.models import EfficientNetClassifier, PredictionOutput
from app.exception_handler import validation_exception_handler, python_exception_handler

import torch


CLASS_NAMES = ["fit", "overweight"]

app = FastAPI()

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

model = EfficientNetClassifier(targets=CLASS_NAMES)


@app.on_event("startup")
async def startup_event():

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    model.load_model()

    # add model to app state
    app.package = {
        "model": model
    }


async def get_classification(raw_image: UploadFile) -> PredictionOutput:
    image_tensor = model.transform_image(raw_image.file.read())

    output = model.model(image_tensor)
    print("Output: ", output)
    predicted_class = torch.max(output, 1).indices.item()
    print("Predicted class:", predicted_class)
    category = model.targets[predicted_class]
    return PredictionOutput(category=category)


@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(get_classification)
) -> PredictionOutput:
    return output
