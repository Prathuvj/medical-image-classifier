import os
import fitz  
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO
import requests

def extract_from_pdf(pdf_path, output_dir="temp_images"):
    """
    Extracts embedded images from a PDF using PyMuPDF (fitz).
    Saves them to the output_dir and returns the list of image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    paths = []
    count = 0
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            img_path = os.path.join(output_dir, f"pdf_image_{count}.{image_ext}")
            image.save(img_path)
            paths.append(img_path)
            count += 1

    return paths


def extract_from_url(url, output_dir="temp_images"):
    """
    Downloads all image files from a webpage and saves them to output_dir.
    Returns list of saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to fetch URL: {e}")
        return []

    img_tags = soup.find_all("img")
    paths = []
    for i, img_tag in enumerate(img_tags):
        src = img_tag.get("src")
        if not src:
            continue

        if not src.startswith("http"):
            src = requests.compat.urljoin(url, src)

        try:
            img_data = requests.get(src).content
            image = Image.open(BytesIO(img_data)).convert("RGB")
            img_path = os.path.join(output_dir, f"url_image_{i}.jpg")
            image.save(img_path)
            paths.append(img_path)
        except Exception:
            continue

    return paths