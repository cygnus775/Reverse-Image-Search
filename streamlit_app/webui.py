import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000/api/upload_image"
BASE_URL = "http://localhost:8000"


def get_images(url: str, upload_image: bytes, num_images: int) -> list:
    with st.spinner("Searching..."):
        # To read file as bytes:
        bytes_data = upload_image.getvalue()
        headers = {
            'accept': 'application/json',
            # requests won't add a boundary if this header is set when you pass files=
            # 'Content-Type': 'multipart/form-data',
        }

        params = {
            'num_images': num_images,
        }

        files = {
            'upload_file': bytes_data,
        }

        response = requests.post(url, params=params, headers=headers, files=files)
        image_list = response.json()["message"]
        return image_list


st.set_page_config(page_title="Reverse Image Search", page_icon=None, layout="centered", initial_sidebar_state="auto",
                   menu_items=None)

st.header("Reverse Image Search")
st.text("Upload and image to find similar images.")

upload_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
num_img = st.number_input("Enter the number of images to find", min_value=1, max_value=100, value=10)

if st.button("Start Search"):
    if upload_img is not None:
        images = get_images(url=API_URL, upload_image=upload_img, num_images=num_img)
        static_image_links = [BASE_URL + image_link for image_link in images]

        for image_link in static_image_links:
            img = requests.get(image_link)
            if img.ok:
                image_content = Image.open(BytesIO(img.content))
                st.image(image_content, use_column_width='auto')

    else:
        st.error("Please provide an image")
