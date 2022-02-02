from importlib_metadata import version
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms as T
import torch.optim as optim

from sklearn.metrics import accuracy_score, confusion_matrix

from model import build_efficientnet

BATCH_SIZE = 32
EPOCHS = 10

img_transforms = {
    'train':
    T.Compose([
        T.RandomResizedCrop(size=(224, 224)),
        T.ToTensor(),
        T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3),
        T.RandomRotation(degrees=30),
        T.RandomHorizontalFlip(p=0.4),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),

    'validation':
    T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),

    'test':
    T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

data_path = "data"
train_path = os.path.join(data_path, "train")
validation_path = os.path.join(data_path, "validation")
test_path = os.path.join(data_path, "test")


train_files = datasets.ImageFolder(
    train_path, transform=img_transforms["train"])
validation_files = datasets.ImageFolder(
    validation_path, transform=img_transforms["validation"])
test_files = datasets.ImageFolder(test_path, transform=img_transforms["test"])

loaders = {
    "train": DataLoader(train_files, batch_size=BATCH_SIZE, shuffle=True),
    "validation": DataLoader(validation_files, batch_size=BATCH_SIZE, shuffle=True),
    "test": DataLoader(test_files, batch_size=BATCH_SIZE, shuffle=True)
}

model = build_efficientnet(version=0)

use_cuda = torch.cuda.is_available()

if use_cuda:
    model = model.cuda()

# Plot model summary
# summary(model, (3, 224, 224))

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Training the model


def train(model, loaders, epochs, optimizer, loss_function, use_cuda, save_path):
    """Returns a trained model and a list of trainning and validation losses."""

    min_validation_loss = np.Inf
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        train_loss = .0
        validation_loss = .0

        # Trainning
        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        # Validation
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders["validation"]):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)
            validation_loss += ((1 / (batch_idx + 1)) *
                                (loss.data - validation_loss))

        train_loss /= len(train_files)
        validation_loss /= len(validation_files)

        training_losses.append(train_loss.cpu().numpy())
        validation_losses.append(validation_loss.cpu().numpy())

        print(
            f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {validation_loss:.6f}')

        if validation_loss < min_validation_loss:
            torch.save(model.state_dict(), save_path)
            min_validation_loss = validation_loss

    return model, training_losses, validation_losses


model, training_losses, validation_losses = train(
    model, loaders, EPOCHS, optimizer, loss_function, use_cuda, save_path="model.pt")

# Plotting the training loss
fig = plt.figure()
plt.plot(range(EPOCHS), training_losses)
plt.title("training loss")
plt.xlabel('Epochs')
plt.ylabel("loss")
fig.savefig("results/training_loss.png")

# Plotting the validation loss
fig = plt.figure()
plt.plot(range(EPOCHS), validation_losses)
plt.title("Validation loss")
plt.xlabel('Epochs')
plt.ylabel("loss")
plt.savefig("results/validation_loss.png")

# Testing the model


def test(model, loader, loss_function, use_cuda):
    test_loss = .0
    correct = .0
    total = .0
    preds = []
    targets = []

    model.eval()

    for batch_idx, (data, target) in enumerate(loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = loss_function(output, target)
        test_loss += ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # converting the output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        preds.append(pred)
        targets.append(target)
        # compare predictions
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    return preds, targets


preds, targets = test(model, loaders["test"], loss_function, use_cuda)

# converting the tensor object to a list for metric functions
preds2, targets2 = [], []

for i in preds:
    for j in range(len(i)):
        preds2.append(i.cpu().numpy()[j])
for i in targets:
    for j in range(len(i)):
        targets2.append(i.cpu().numpy()[j])

# Computing the accuracy
acc = accuracy_score(targets2, preds2)
print("Accuracy: ", acc)

cm = confusion_matrix(targets2, preds2)

fig = plt.figure()
plt.matshow(cm)
plt.title('Confusion Matrix Dog Classification')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicated')
plt.savefig("results/confusion_matrix.png")

# Finally, call the matplotlib show() function to display the visualization
plt.show()
