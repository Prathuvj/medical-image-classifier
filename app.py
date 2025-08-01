import io
import uuid
import requests
from flask import Flask, request, jsonify
from PIL import Image
from extract_images import extract_from_pdf
from predict import classify_image
import fitz

app = Flask(__name__)

ALLOWED_IMAGE_TYPES = {"jpg", "jpeg", "png"}
ALLOWED_PDF_TYPE = "pdf"

def infer_from_image(image):
    """Run prediction on a single PIL image object"""
    image = image.convert("RGB")
    temp_path = f"in-memory-{uuid.uuid4()}.jpg"
    image.save(temp_path)
    label, confidence = classify_image(temp_path)
    return {
        "image": temp_path,
        "prediction": label,
        "confidence": round(confidence, 4)
    }

def infer_from_pdf_bytes(file_bytes):
    """Extract and classify images from PDF bytes (in memory)"""
    results = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            result = infer_from_image(image)
            result["image"] = f"pdf_page{page_index}_img{img_index}.jpg"
            results.append(result)

    return results

@app.route("/predict", methods=["POST"])
def predict():
    results = []

    if "file" in request.files:
        file = request.files["file"]
        filename = file.filename.lower()

        if filename.endswith(ALLOWED_PDF_TYPE):
            file_bytes = file.read()
            results = infer_from_pdf_bytes(file_bytes)
        elif any(filename.endswith(ext) for ext in ALLOWED_IMAGE_TYPES):
            image = Image.open(file.stream)
            results = [infer_from_image(image)]
        else:
            return jsonify({"error": "Unsupported file type"}), 400

    elif "url" in request.json:
        url = request.json["url"]
        try:
            response = requests.get(url)
            content_type = response.headers.get("Content-Type", "")
            content = response.content

            if "pdf" in content_type or url.lower().endswith(".pdf"):
                results = infer_from_pdf_bytes(content)
            elif "image" in content_type or any(url.lower().endswith(ext) for ext in ALLOWED_IMAGE_TYPES):
                image = Image.open(io.BytesIO(content))
                results = [infer_from_image(image)]
            else:
                return jsonify({"error": "Unsupported file type in URL"}), 400

        except Exception as e:
            return jsonify({"error": f"Failed to fetch or process URL: {str(e)}"}), 400
    else:
        return jsonify({"error": "No file or URL provided"}), 400

    return jsonify({"status": "success", "results": results})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)