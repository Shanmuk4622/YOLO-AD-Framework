
# 🚀 YOLO Anomaly Detection Framework  

## 📌 Overview  
This repository provides a comparative study and implementation of multiple **YOLO architectures** (YOLOv5, YOLOv8, etc.) to build a robust framework for **anomaly detection in video streams**.  
Developed during the **Hyperland National Hackathon**, the project showcases real-time computer vision innovation for detecting irregular or unsafe activities.  



## 📂 Repository Structure  

```
YOLO-AD-Framework/
├── README.md                 # Project overview and usage instructions
├── LICENSE                   # MIT License
├── requirements.txt           # Dependencies for local inference
├── paper/
│   ├── paper.tex              # XeLaTeX-formatted research paper
│   └── paper.pdf              # Compiled PDF
├── code/
│   └── main.py                # Core training and evaluation logic
├── models/                    # Trained YOLO models (.pt files)
├── data/                      # Datasets and sample videos
│   ├── sample.csv             # (Optional) Sample annotations
│   └── README.md              # Dataset usage instructions
├── results/
│   └── output_graphs.png      # Performance metrics visualization
├── inference/
│   ├── detect.py              # YOLO inference script
│   └── test.mp4               # Example input video
```


## ⚙️ Setup Instructions  

### 🔧 Training (Google Colab)  
- No local installation required.  
- Open the provided Colab notebook and follow the structured steps.  
- Choose YOLOv5 or YOLOv8 models within the notebook.  

### 🧩 Dataset Preparation  

**Option 1: Custom Dataset**  
- Compress your dataset folder as `.zip`.  
- Upload it to Google Drive.  

**Option 2: Sample Dataset**  
- Download and upload this dataset to your Drive:  
  🔗 [Download Videos](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&dl=0)  

Ensure your dataset follows YOLO format with a `data.yaml`:  
```yaml
train: path/to/train/images
val: path/to/val/images

nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

---

## 🧠 Model Training (Colab)  

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Choose YOLO variant
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

> Recommended: Python 3.8  

1. Install dependencies:  
```bash
pip install -r requirements.txt
```

2. Run inference:  
```bash
python inference/detect.py --weights models/best.pt --source inference/test.mp4
```

### Sample Inference Script (`detect.py`)  
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
Refer to `data/README.md` for performance comparisons, sample videos, and evaluation metrics.  

---

## 🏆 Hackathon Context  
This framework was built during the **Hyperland National Hackathon**, emphasizing teamwork, rapid prototyping, and research-driven innovation in anomaly detection.  

---

## 📝 License  
 ## 📝 License  
This project is currently **unlicensed** and shared as a personal side project.  
You are welcome to explore, learn from, and adapt the code for your own experiments.  

Please note:  
- ⚠️ There is **no warranty** or guarantee of fitness for any purpose.  
- ⚠️ Use at your own risk — the author is not responsible for any issues or damages.  
- ✅ Feel free to fork or modify for educational and research purposes.  
- ❌ Commercial use or redistribution is not permitted without explicit permission.  

If you plan to build upon this work or use it beyond personal learning, please reach out to discuss appropriate licensing.  


