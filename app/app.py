from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.logger import logger
from app.config import CONFIG
from app.models import EfficientNetClassifier, PredictionOutput

CLASS_NAMES = ["fit", "overweight"]

app = FastAPI()

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


# @app.post("/prediction")
# async def get_image_prediction(image: UploadFile) -> PredictionOutput:
#     # prediction = app.package["model"].get_classification(image.file.read())
#     prediction = model.get_classification(image.file.read())
#     print(prediction)
#     if image:
#         return prediction


# TODO: Implement /prediction as dependency injection 
@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(model.get_classification)
) -> PredictionOutput:
    return output
