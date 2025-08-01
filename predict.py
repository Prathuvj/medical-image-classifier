import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from config import *

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

import os
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("best_resnet_medical_classifier.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def classify_image(img_path, threshold=THRESHOLD):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        max_prob, pred_class = torch.max(probs, dim=1)

    confidence = max_prob.item()
    label = class_names[pred_class.item()]

    if confidence < threshold:
        return "non-medical", confidence
    else:
        return f"medical ({label})", confidence

if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    prediction, confidence = classify_image(img_path)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")