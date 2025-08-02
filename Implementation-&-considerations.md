# Implementation & Considerations

To classify images as medical or non-medical, I trained a supervised classifier on four distinct medical imaging modalities:

- **X-ray**
- **MRI**
- **CT Scan**
- **Ultrasound**

## Key Design Decisions

### Medical-only Classifier + Confidence Threshold
Instead of explicitly training on non-medical images (which are diverse and undefined), I trained a model only on medical classes and used a confidence threshold during inference.

- **If the model confidently predicts one of the four known medical types** → classify as "medical"
- **Otherwise** → "non-medical"

### Model Choice: torchvision ResNet-50
- Lightweight and well-suited for transfer learning
- Pretrained on ImageNet for robust feature extraction
- Fine-tuned on the 4-class medical dataset

### No Hardcoding
Images are extracted dynamically from PDFs or URLs using PyMuPDF and BeautifulSoup, allowing real-world adaptability.

### User Flexibility
- Supports both API (via Flask) and GUI (via Streamlit) access
- Allows uploads or remote testing via URL, useful for automation or manual testing

## Performance / Efficiency Considerations

| Metric | Consideration |
|--------|---------------|
| **Inference Speed** | ResNet-50 offers fast inference even on CPU, suitable for real-time API use |
| **Model Size** | ~97 MB (ResNet-50), balances accuracy with deployability |
| **Threshold Optimization** | Tuned softmax confidence threshold (~0.85) to reduce false positives |
| **No File Storage** | Files are processed in-memory for ideal performance |
| **Concurrent Access** | Flask is multi-threaded; Streamlit UI runs independently for parallel testing |
| **Data Simplification** | Only medical images are used for training, so there are no noisy or ambiguous negatives |

## Technical Implementation Details

### Confidence Threshold Strategy
The model uses a confidence-based approach to distinguish between medical and non-medical images:

```python
def classify_image(img_path, threshold=THRESHOLD):
    # ... model inference ...
    confidence = max_prob.item()
    label = class_names[pred_class.item()]
    
    if confidence < threshold:
        return "non-medical", confidence
    else:
        return f"medical ({label})", confidence
```

### In-Memory Processing
All file processing happens in memory without creating temporary files:
- PDF images are extracted directly to PIL Image objects
- URL content is fetched and processed in BytesIO streams
- No disk I/O overhead for temporary file management

### Multi-Modal Support
The system handles various input formats seamlessly:
- **Single Images**: Direct classification
- **PDF Documents**: Automatic image extraction and batch processing
- **Web URLs**: Dynamic content fetching and processing

## Architecture Benefits

### Scalability
- Stateless design allows horizontal scaling
- In-memory processing reduces I/O bottlenecks
- Concurrent request handling via Flask threading

### Security
- No temporary file storage eliminates security risks
- Input validation prevents malicious file uploads
- URL processing includes timeout and size limits

### Maintainability
- Clean separation between extraction, inference, and API layers
- Configurable parameters via `config.py`
- Modular design allows easy component updates

## Future Enhancements

- **GPU Acceleration**: CUDA support for faster inference
- **Model Ensemble**: Combine multiple models for better accuracy
- **Advanced Thresholding**: Dynamic threshold adjustment based on image characteristics
- **Caching**: Redis integration for frequently accessed results
- **Monitoring**: Comprehensive logging and metrics collection