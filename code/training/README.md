# 🚀 YOLO Model Training Guide (Google Colab Ready)

This README provides a clean and universal guide for training a YOLO models on a custom dataset using **Google Colab**.

---

## ⚙️ Setup Instructions (Colab Only)

- No need to install dependencies locally.
- All installations and setups will be handled inside the Colab notebook.

Simply open the training notebook and follow the step-by-step instructions.

---

## 📦 Dataset Preparation

### Option 1: Using Your Own Dataset
- Compress your dataset folder as a `.zip` file.
- Upload it to your Google Drive.

### Option 2: Use Sample Dataset
Download and upload the dataset from this link to your Google Drive:  
🔗 [Download Sample Dataset](https://drive.google.com/file/d/12gw_WPnWrmo17kHacBnJQfek72GT0Ac1/view?usp=drive_link)

Prepare your dataset in the following YOLO format and save it as `data.yaml`:

```yaml
train: path/to/train/images
val: path/to/val/images

nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

---

## 🧠 Model Training

1. **Select YOLO model variant** (e.g., `yolov8n.pt`, `yolov5s.pt`, etc.)
2. **Train the model inside Colab using the following code:**

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or any other variant
model.train(data="data.yaml", epochs=100)
```

- After training, the best weights will be saved at:
```
runs/detect/train/weights/best.pt
```

You can use this trained model later in your local environment for detection and evaluation.

---

## 📝 License

This training setup is part of an open-source research project licensed under the [MIT License](LICENSE).

