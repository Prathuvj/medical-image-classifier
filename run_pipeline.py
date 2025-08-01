# run_pipeline.py

import os
import argparse
from predict import classify_image
from extract_images import extract_from_pdf, extract_from_url

def run_pipeline(input_path):
    if input_path.startswith("http"):
        print(f"\n🔗 Extracting images from URL: {input_path}")
        image_paths = extract_from_url(input_path)
    elif input_path.endswith(".pdf"):
        print(f"\n📄 Extracting images from PDF: {input_path}")
        image_paths = extract_from_pdf(input_path)
    else:
        print("❌ Unsupported input. Use a PDF or URL.")
        return

    if not image_paths:
        print("⚠️ No images found.")
        return

    print(f"\n🧠 Running predictions on {len(image_paths)} images...\n")
    for path in image_paths:
        prediction, confidence = classify_image(path)
        print(f"{os.path.basename(path)} → {prediction} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="PDF path or URL")
    args = parser.parse_args()

    run_pipeline(args.input)