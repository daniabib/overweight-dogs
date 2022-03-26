import requests

import streamlit as st
import boto3

from image_processing import process_image


BACKEND = "http://backend:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']


def detect_labels_rekognition(image):
    client = boto3.client("rekognition")

    response = client.detect_labels(Image={"Bytes": image})

    is_dog = False
    detected_label: str
    confidence = 0

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


def classify(image: bytes, server_url: str) -> str:
    r = requests.post(
        server_url, files={"raw_image": ("filename", image, "image/jpeg")},
        timeout=8000)
    print(r.json())

    return r.json()["category"]


def rekognition_string_process(detected_label):
    an_rule = ["a", "e", "i", "o", "u"]

    return f"an {detected_label.lower()}" if detected_label.lower()[0] in an_rule \
        else f"a {detected_label.lower()}"


# construct UI layout
st.title("Overweight Dogs Classifier")

st.header("Is my dog overweight?")

st.write("This is a simple application that tells by the image of your dog if it is overweight.")

input_image = st.file_uploader("Insert image", type=SUPPORTED_TYPES)

if st.button("Upload your dog image"):
    if input_image:
        bytes_image = input_image.getvalue()
        output_image = process_image(bytes_image)

        st.image(output_image, caption='Uploaded Image.',
                 use_column_width=True)
        st.write("")

        is_dog, detected_label, confidence = detect_labels_rekognition(
            output_image)

        if is_dog:
            classification = classify(output_image, BACKEND)
            st.header(f"Your dog seens {classification}.")
        else:
            st.header(
                f"Are you sure it's a dog? It seems more like {rekognition_string_process(detected_label)}.")
    else:
        st.write("Insert an image!")

st.write("")
st.subheader("Disclaimer")
st.write("This application is just an exercise on Deep Learning, Computer Vision, and Model Deployment. It has no intention of having any scientific validation. Please, use it only as a code example.")
