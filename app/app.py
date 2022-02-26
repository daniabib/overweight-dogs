from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.logger import logger
from app.config import CONFIG
from app.models import EfficientNetClassifier, PredictionOutput
import torch


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
#     # prediction = model.get_classification(image.file.read())
#     prediction = model.get_classification(image)
#     print(prediction)
#     if image:
#         return prediction
async def get_classification(raw_image: UploadFile) -> PredictionOutput:
    image_tensor = model.transform_image(raw_image.file.read())

    output = model.model(image_tensor)
    print("Output: ", output)
    predicted_class = torch.max(output, 1).indices.item()
    print("Predicted class:", predicted_class)
    category = model.targets[predicted_class]
    return PredictionOutput(category=category)

# TODO: Implement /prediction as dependency injection


@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(get_classification)
) -> PredictionOutput:
    return output
