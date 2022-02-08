import io

from fastapi import FastAPI, File, UploadFile
from fastapi.logger import logger
from app.config import CONFIG
from app.classification import get_model, transform_image, get_classification

app = FastAPI()


@app.on_event("startup")
async def startup_event():

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    model = get_model()

    # add model to app state
    app.package = {
        "model": model
    }

@app.post("/predict")
async def get_image_prediction(image: UploadFile):
    prediction = get_classification(app.package["model"], image.file.read())
    print(prediction)
    if image:
        return {"predicted_class": prediction}
