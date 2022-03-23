import requests
from PIL import Image
from PIL.ExifTags import TAGS
import streamlit as st

BACKEND = "http://backend:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']


def process_image(image) -> Image:
    """Transform a BytesIO object into a PIL Image.
       Also handle image orientation."""
    image = Image.open(image)
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

    return image


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

input_image = st.file_uploader(
    "insert image", type=SUPPORTED_TYPES)

if st.button("Upload your dog image"):
    if input_image:
        bytes_image = input_image.getvalue()
        output_image = process_image(input_image)
        st.image(output_image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        classification = classify(bytes_image, BACKEND)
        st.header(f"Your dog seens {classification}.")
    else:
        st.write("Insert an image!")

st.subheader("Disclaimer")
st.write("This application is just an exercise on Deep Learning, Computer Vision, and Model Deployment. It has no intention of having any scientific validation. Please, use it only as a code example.")
