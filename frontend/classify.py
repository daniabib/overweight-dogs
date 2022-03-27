from typing import Tuple
import requests

import boto3


def detect_labels_rekognition(image: bytes) -> Tuple[bool, str, float]:
    client = boto3.client("rekognition")

    response = client.detect_labels(Image={"Bytes": image})

    is_dog = False
    detected_label: str
    confidence = 0

    # Print Labels for debugging
    print('Detected labels for ')
    print()
    for label in response['Labels']:
        print("Label: " + label['Name'])
        print("Confidence: " + str(label['Confidence']))
        print("Instances:")
        for instance in label['Instances']:
            print("  Bounding box")
            print("    Top: " + str(instance['BoundingBox']['Top']))
            print("    Left: " + str(instance['BoundingBox']['Left']))
            print("    Width: " + str(instance['BoundingBox']['Width']))
            print("    Height: " + str(instance['BoundingBox']['Height']))
            print("  Confidence: " + str(instance['Confidence']))
            print()

        print("Parents:")
        for parent in label['Parents']:
            print("   " + parent['Name'])
        print("----------")
        print()

        if label["Name"] == "Dog":
            is_dog = True
            detected_label = label["Name"]
            confidence = label["Confidence"]
            break

        if label["Confidence"] > confidence:
            detected_label = label["Name"]
            confidence = label["Confidence"]

    return is_dog, detected_label, confidence


def classify_dogs(image: bytes, server_url: str) -> str:
    r = requests.post(
        server_url, files={"raw_image": ("filename", image, "image/jpeg")},
        timeout=8000)
    print(r.json())

    return r.json()["category"]


def rekognition_string_process(detected_label: str) -> str:
    an_rule = ["a", "e", "i", "o", "u"]

    return f"an {detected_label.lower()}" if detected_label.lower()[0] in an_rule \
        else f"a {detected_label.lower()}"