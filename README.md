# 🚀 YOLO Anomaly Detection Framework

This repository presents a comparative study and implementation of various YOLO architectures to build a robust framework for anomaly detection in video streams.

---

## 📁 Repository Structure

```
your-project/
├── README.md                 # Project overview and usage instructions
├── LICENSE                   # MIT License
├── requirements.txt          # Dependencies (for local inference)
├── paper/
│   ├── your_paper.tex        # XeLaTeX-formatted research paper
│   └── your_paper.pdf        # Compiled PDF
├── code/
│   └── main.py               # Core implementation (training and evaluation logic)
├── models/                   # Trained YOLO models (.pt files)
├── data/                     # Datasets and sample videos
│   └── sample.csv            # (Optional) Sample annotations
│   └── README.md             # Instructions to download and use the dataset
├── results/
│   └── output_graphs.png     # Visual performance metrics
├── inference/
│   ├── detect.py             # YOLO inference script
│   └── test.mp4              # Example input video for detection
```

---

## ⚙️ Setup Instructions

### 🔧 Training (Google Colab)
- No need to install dependencies locally.
- Open the provided Colab notebook and follow the structured steps.
- YOLOv5/v8 models can be selected within the notebook.

### 🧩 Dataset Preparation

#### Option 1: Use Your Own Dataset
- Compress your dataset folder as a `.zip` file.
- Upload it to your Google Drive.

#### Option 2: Download Sample Dataset
- Download and upload this dataset to your Drive:  
  🔗 [Download Videos](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&dl=0)

Ensure your dataset follows the YOLO format and define a corresponding `data.yaml`:
```yaml
train: path/to/train/images
val: path/to/val/images

nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

### 🧠 Model Training (Colab)
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Choose your YOLO variant
data_yaml = "path/to/data.yaml"
model.train(data=data_yaml, epochs=100)
```
Output:
```
runs/detect/train/weights/best.pt
```
Use this `.pt` file for inference.

---

## 🎯 Model Inference (Local Machine)
(we recommend you to use Python 3.8)
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model:
```bash
python inference/detect.py --weights models/best.pt --source inference/test.mp4
```

### Sample Inference Code (`detect.py`):
```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict(source="inference/test.mp4", save=True)
```
Outputs will be saved in:
```
runs/detect/predict/
```

---

## 📊 Performance Metrics
Refer to `data/README.md` for model performance comparisons, sample videos, and metrics.

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).


