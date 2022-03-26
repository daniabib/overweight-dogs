import string
import requests
from PIL import Image
from PIL.ExifTags import TAGS

import streamlit as st

import io
import boto3

BACKEND = "http://backend:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


def process_image(image) -> Image:
    """Transform a BytesIO object into a PIL Image.
       Also handle image orientation."""
    # image = Image.open('nico.jpg')
    image_byte_size = len(image)
    image = Image.open(io.BytesIO(image))

    orientation = None
    for key, value in image.getexif().items():
        if TAGS.get(key) == "Orientation":
            orientation = value

    if orientation == 3:
        image = image.transpose(Image.ROTATE_180)
    if orientation == 6:
        image = image.transpose(Image.ROTATE_270)
    if orientation == 8:
        image = image.transpose(Image.ROTATE_90)


    # TODO: Resize image if its over rekognition size limit
    if image_byte_size >= 5042880:
        width, height = image.size

        print("Image size in bytes before resize: ", format_bytes(image_byte_size))
        image = image.resize((int(width*.4), int(height*.4)), resample=Image.ANTIALIAS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=40)
        # image.close()
        image_bytes = buffer.getvalue()

        # image = Image.open(io.BytesIO(image_bytes))
        # image_bytes2 = image.tobytes()
        print("Image size in bytes(2) AFTER resize: ", format_bytes(len(image_bytes)))
        return image_bytes

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=40)
    # image.close()
    image_bytes = buffer.getvalue()
    
    return image_bytes


def detect_labels_rekognition(image):
    client = boto3.client("rekognition", )

    response = client.detect_labels(Image={"Bytes": image})

    is_dog = False
    detected_label: string
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


# construct UI layout
st.title("Overweight Dogs Classifier")

st.header("Is my dog overweighted?")

st.write("This is simple application that tells by the image of your dog if it is overweighted.")

input_image = st.file_uploader("Insert image", type=SUPPORTED_TYPES)

if st.button("Upload your dog image"):
    if input_image:
        bytes_image = input_image.getvalue()
        output_image = process_image(bytes_image)
        # output_image = process_image(input_image.getvalue())
        # print("Type bytes_image: ", type(bytes_image))
        print("Byte size output_im1: ", format_bytes(len(output_image)))

        # print(output_image)
        st.image(output_image, caption='Uploaded Image.',
                 use_column_width=True)
        st.write("")

        # output_image = output_image.tobytes()
        
        print("Type output_image: ", type(output_image))
        print("Byte size output_im2: ", format_bytes(len(output_image)))

        is_dog, detected_label, confidence  = detect_labels_rekognition(output_image)
        # is_dog, detected_label, confidence  = detect_labels_rekognition(bytes_image)
        # detected_label = "TEST: REKOGNITION"
        # is_dog = True
        if is_dog:
            # classification = classify(output_image, BACKEND)
            classification = "TEST"
            st.header(f"Your dog seens {classification}.")
        
        else:
            st.header(f"Are you sure it's a dog? If seens more like a {detected_label}.")
        

    else:
        st.write("Insert an image!")

st.write("")
st.subheader("Disclaimer")
st.write("This application is just an exercise on Deep Learning, Computer Vision, and Model Deployment. It has no intention of having any scientific validation. Please, use it only as a code example.")
