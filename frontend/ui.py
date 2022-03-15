import io
from tarfile import SUPPORTED_TYPES

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with backend endpoint
# backend = "http://backend:8080/prediction"
BACKEND = "http://0.0.0.0:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']

def process(image, server_url: str):

    # m = MultipartEncoder(fields={"raw_image": ("filename",
    #      image, "image/jpeg")})

    r = requests.post(
        server_url, files={"raw_image": ("filename", image, "image/jpeg")}, timeout = 8000
    )

    return r.json()["category"]


# construct UI layout
st.title("Overweight Dogs Classifier")

st.write(
    """Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8080/docs` for FastAPI documentation."""


)  # description and instructions

input_image=st.file_uploader("insert image", type=SUPPORTED_TYPES)  # image upload widget

if st.button("Upload your dog image:"):

    if input_image:
        # print(input_image)
        # input_image = Image.open(input_image).convert("RGB")
        # print(input_image)
        # st.image(original_image, caption='Uploaded Image.', use_column_width=True)
        st.image(input_image, caption = 'Uploaded Image.',
                 use_column_width = True)
        st.write("")
        st.write("Classifying...")
        st.write("")
        classification=process(input_image, BACKEND)
        st.write(f"Your dog seens {classification}")

    else:
        # handle case with no image
        st.write("Insert an image!")
