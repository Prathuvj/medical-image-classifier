import os
from predict import classify_image
import csv

TEST_DIR = "test_images"
OUTPUT_CSV = "test_results.csv"

results = []

print(f"🧪 Testing all images in: {TEST_DIR}\n")

for filename in os.listdir(TEST_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(TEST_DIR, filename)
        label, confidence = classify_image(path)

        print(f"{filename:<25} → {label:<15} (Confidence: {confidence:.2f})")
        results.append([filename, label, round(confidence, 4)])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Prediction", "Confidence"])
    writer.writerows(results)

print(f"\n✅ Results saved to {OUTPUT_CSV}")