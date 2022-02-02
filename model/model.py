import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def build_efficientnet(version: int = 0):
    model = EfficientNet.from_pretrained(f"efficientnet-b{version}")

    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    in_features = model._fc.in_features

    # Replace the top layer with our own
    model._fc = nn.Sequential(
        nn.BatchNorm1d(num_features=in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )

    return model


if __name__ == "__main__":
    model = build_efficientnet()
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
