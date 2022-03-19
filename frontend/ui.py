import requests

import streamlit as st

BACKEND = "http://backend:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']


def process(image, server_url: str):
    r = requests.post(
        server_url, files={"raw_image": ("filename", image, "image/jpeg")},
        timeout=8000)

    return r.json()["category"]


# construct UI layout
st.title("Overweight Dogs Classifier")

st.write(
    """Is my dog overweighted?.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8080/docs` for FastAPI documentation."""
)

input_image = st.file_uploader(
    "insert image", type=SUPPORTED_TYPES)  # image upload widget

if st.button("Upload your dog image"):

    if input_image:
        st.image(input_image, caption='Uploaded Image.',
                 use_column_width=True)
        st.write("")
        classification = process(input_image, BACKEND)
        st.write(f"Your dog seens {classification}.")

    else:
        # handle case with no image
        st.write("Insert an image!")
