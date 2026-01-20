import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to identify the disease.")

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing leaf..."):
            files = {
                "img": (   # 🔑 must match FastAPI parameter name
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success(
                        f"🌿 **Disease:** {result['predicted_class']}\n\n"
                        f"📊 **Confidence:** {result['confidence']}"
                    )
                else:
                    st.error(f"API Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ FastAPI server is not running")
