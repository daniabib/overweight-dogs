import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torchvision import datasets
from model import build_efficientnet
import matplotlib.pyplot as plt


BATCH_SIZE = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

img_transforms = {
    'validation':
    T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

data_path = "data"
validation_path = os.path.join(data_path, "validation")


validation_files = datasets.ImageFolder(
    validation_path, transform=img_transforms["validation"])

loaders = {
    "validation": DataLoader(validation_files, batch_size=BATCH_SIZE, shuffle=True),
}


model = build_efficientnet()
model.load_state_dict(torch.load(
    "model.pt", map_location=torch.device(device)
))
model = model.to(device)
model.eval()

class_names = validation_files.classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(loaders['validation']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(loaders['validation']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
            
#         model.train(mode=was_training)

# visualize_model(model)

test_input = torch.randn((10, 3, 224, 224)).to(device)

preds = model(test_input)

what, y_hat = preds.max(1)
print(what)
print(y_hat)
# preds_classes = torch.max(preds, 1)
# print(preds_classes.indices)
print(class_names)
# print(inputs.shape)
# print(inputs[0])
# print(classes.shape)
# print(classes)