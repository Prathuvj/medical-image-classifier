import os
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from extract_images import extract_from_pdf
from predict import classify_image

app = Flask(__name__)
UPLOAD_FOLDER = "temp_images"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    filename = secure_filename(file.filename)
    file_ext = filename.rsplit(".", 1)[1].lower()
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_ext}")
    file.save(temp_path)

    results = []

    try:
        if file_ext == "pdf":
            image_paths = extract_from_pdf(temp_path, output_dir=UPLOAD_FOLDER)
        else:
            image_paths = [temp_path]

        for path in image_paths:
            label, confidence = classify_image(path)
            results.append({
                "image": os.path.basename(path),
                "prediction": label,
                "confidence": round(confidence, 4)
            })

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)