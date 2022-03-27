import streamlit as st

from image_processing import process_image
from classify import detect_labels_rekognition, classify_dogs
from classify import rekognition_string_process


BACKEND = "http://backend:8080/prediction"
SUPPORTED_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tga', 'tiff']


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
            classification = classify_dogs(output_image, BACKEND)
            st.header(f"Your dog seems {classification}.")
        else:
            st.subheader(
                f"Are you sure it's a dog? It seems more like {rekognition_string_process(detected_label)}.")
    else:
        st.write("Insert an image!")

st.write("")
st.subheader("Disclaimer")
st.write("This application is just an exercise on Deep Learning, Computer Vision, and Model Deployment. It has no intention of having any scientific validation. Please, use it only as a code example.")
