import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
from config import *

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=NUM_CLASSES)
model.load_state_dict(torch.load("resnet_medical_classifier.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

def classify_image(img_path, threshold=THRESHOLD):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        max_prob, pred_class = torch.max(probs, dim=1)
    
    if max_prob.item() < threshold:
        return "non-medical"
    else:
        return "medical"

if __name__ == "__main__":
    path = input("Enter image path: ")
    result = classify_image(path)
    print(f"Prediction: {result}")