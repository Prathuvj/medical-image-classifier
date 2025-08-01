import streamlit as st
from PIL import Image
import io
import requests
import fitz
from extract_images import extract_from_pdf
from predict import classify_image

st.set_page_config(page_title="Medical Image Classifier", layout="centered")
st.title("🧠 Medical vs. Non-Medical Image Classifier")
st.markdown("Upload a PDF or Image file, or enter an image/PDF URL to detect if it's medical or not.")

def infer_from_image(image):
    """Classify a PIL image"""
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    temp_path = f"in-memory-image.jpg"
    image.save(temp_path)
    label, confidence = classify_image(temp_path)
    return label, confidence

def infer_from_pdf(file_bytes):
    """Extract images from PDF and classify each"""
    results = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            label, confidence = infer_from_image(image)
            results.append((image, label, confidence))
    return results

upload_type = st.radio("Choose input method:", ["📤 Upload File", "🔗 Enter URL"])

if upload_type == "📤 Upload File":
    file = st.file_uploader("Upload an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

    if file is not None:
        ext = file.name.lower().split(".")[-1]
        st.info(f"Processing `{file.name}`...")

        if ext == "pdf":
            results = infer_from_pdf(file.read())
            for idx, (image, label, confidence) in enumerate(results):
                st.image(image, caption=f"Prediction: {label} (Confidence: {confidence:.2f})", use_container_width=True)

        elif ext in ["jpg", "jpeg", "png"]:
            image = Image.open(file).convert("RGB")
            label, confidence = infer_from_image(image)
            st.image(image, caption=f"Prediction: {label} (Confidence: {confidence:.2f})", use_container_width=True)

        else:
            st.error("Unsupported file type.")

elif upload_type == "🔗 Enter URL":
    url = st.text_input("Paste the image or PDF URL")

    if url:
        try:
            response = requests.get(url)
            content_type = response.headers.get("Content-Type", "")
            content = response.content

            if "pdf" in content_type or url.endswith(".pdf"):
                results = infer_from_pdf(content)
                for idx, (image, label, confidence) in enumerate(results):
                    st.image(image, caption=f"Prediction: {label} (Confidence: {confidence:.2f})", use_container_width=True)

            elif "image" in content_type or any(url.endswith(ext) for ext in ["jpg", "jpeg", "png"]):
                image = Image.open(io.BytesIO(content)).convert("RGB")
                label, confidence = infer_from_image(image)
                st.image(image, caption=f"Prediction: {label} (Confidence: {confidence:.2f})", use_container_width=True)

            else:
                st.error("URL must point to a valid image or PDF.")

        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")