from fastapi import FastAPI
from fastapi.logger import logger
import torch
from app.config import CONFIG
from model.model import build_efficientnet

app = FastAPI()

@app.on_event("startup")
async def startup_event():

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    model = build_efficientnet()
    model.load_state_dict(torch.load(
        CONFIG["MODEL_PATH"], map_location=torch.device(CONFIG["DEVICE"])
    ))
    model.eval()

@app.get("/predict")
async def prediction():
    
    return {"message": "Predicted val!"}