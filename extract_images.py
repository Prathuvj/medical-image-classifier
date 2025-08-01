import os
import requests
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO
from pdf2image import convert_from_path

def extract_from_pdf(pdf_path, output_dir="temp_images"):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200)

    paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"pdf_image_{i}.jpg")
        img.save(img_path, "JPEG")
        paths.append(img_path)
    return paths

def extract_from_url(url, output_dir="temp_images"):
    os.makedirs(output_dir, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

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