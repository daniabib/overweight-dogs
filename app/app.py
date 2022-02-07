import io

from fastapi import FastAPI, File, UploadFile
from fastapi.logger import logger
import torch
from app.config import CONFIG
from app.classification import get_model, transform_image

app = FastAPI()


@app.on_event("startup")
async def startup_event():

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    model = get_model()


@app.post("/predict")
async def get_image_prediction(image: UploadFile):
    transformed_image = transform_image(image.file.read())

    if image:
        return {"image_type": "get image"}
