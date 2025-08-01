# 🧠 Medical vs Non-Medical Image Classifier

This project is a complete end-to-end solution to classify images as **medical** or **non-medical** using a fine-tuned ResNet-50 model.

It supports:
- ✅ Training on multiple medical modalities (X-ray, CT, MRI, Ultrasound)
- ✅ Inference via uploaded images or PDFs
- ✅ API for programmatic access (Flask)
- ✅ Interactive frontend (Streamlit)
- ✅ Postman/URL testing

---

## 🚀 Features

| Component       | Description                                  |
|-----------------|----------------------------------------------|
| 🧠 Model        | ResNet-50 (torchvision), fine-tuned          |
| 📥 Inputs       | PDF, image upload, or image/PDF URL          |
| 🧪 Inference    | Classifies each image as "medical" or "non-medical" |
| 🖥️ Frontend     | Streamlit UI for manual testing              |
| 🧵 Backend      | Flask API for integration or Postman testing |
| 🧰 Tools        | PyTorch, PIL, PyMuPDF, BeautifulSoup, Streamlit, Flask |

---

## ⚙️ Setup Instructions

### 1. Clone the repo & install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train (optional, already trained model provided)
```bash
python train_resnet.py
```

### 3. Start full app (API + UI)
```bash
python start_app.py
```

---

## 🧪 Test the API

### 📤 Via Postman (or curl)

**Endpoint:** `http://127.0.0.1:5000/predict`  
**Method:** POST

**Option 1: Upload file (image or PDF)**  
Use form-data, key = `file`, value = upload file

**Option 2: Provide URL**  
Content-Type: application/json

```json
{
  "url": "https://example.com/sample.pdf"
}
```

---

## 🖥️ Use the Streamlit App

After running:
```bash
python start_app.py
```
Visit:

```
http://localhost:8501
```

There you can:
- Upload images or PDFs
- Enter image/PDF URLs
- View predictions with confidence and preview

---

## 🎯 Model Details

- Based on `torchvision.models.resnet50`
- Trained on 4-class medical data: **X-ray, CT, MRI, Ultrasound**
- Inference uses confidence threshold to detect non-medical images