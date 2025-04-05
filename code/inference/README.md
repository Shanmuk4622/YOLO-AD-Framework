# 🚀 YOLO Model Inference Guide

---
(we recommend you to use Python 3.8)
## ⚙️ Setup Instructions ( Local for Inference)

### 🧩 Inference (Local Setup)
For running inference and evaluating results, install dependencies locally:
```bash
pip install -r requirements.txt
```

---
## 🎯 Model Inference (Local Machine)

1. Place your trained model (`best.pt`) in the `models/` directory.
2. Use the provided `detect.py` file to run inference:
i have changed the model name from best.pt to y10m.pt
`

### detect.py Example Logic:
```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict(source="inference/test.mp4", save=True)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

