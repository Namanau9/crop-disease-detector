# 🌱 AI Crop Disease Detection System

A Deep Learning-based web application that detects plant diseases from leaf images using **PyTorch + EfficientNet-B0** and provides treatment & prevention guidance.

---

## 🚀 Project Overview

This project uses Transfer Learning with EfficientNet-B0 to classify plant leaf diseases across:

- 🌶 Pepper
- 🥔 Potato
- 🍅 Tomato

The system provides:

- ✅ Disease prediction
- 📊 Confidence score
- 📈 Class probability visualization
- 💊 Treatment recommendation
- 🌿 Prevention advice
- 🌐 Interactive Streamlit dashboard

---

## 🧠 Model Details

- Architecture: EfficientNet-B0
- Framework: PyTorch
- Training Strategy:
  - Stage 1: Frozen feature extractor
  - Stage 2: Fine-tuning entire network
- GPU Acceleration: NVIDIA RTX 4050 (CUDA)
- Final Validation Accuracy: ~98%

---

## 📂 Dataset

Dataset used: **PlantVillage Dataset**  
Filtered for Pepper, Potato, and Tomato classes.

Classes include:

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Mosaic Virus
- Yellow Leaf Curl Virus
- Healthy Leaves

---

## 🏗 Project Structure

```
Crop_Disease_PyTorch/
│
├── dataset/
├── models/
│   └── crop_model.pth
├── train.py
├── predict.py
├── app.py
├── disease_info.py
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/crop-disease-detector.git
cd crop-disease-detector
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋 Training the Model

```bash
python train.py
```

The trained model will be saved inside:

```
models/crop_model.pth
```

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

Then open in your browser:

```
http://localhost:8501
```

---

## ⚠ Important Note

This model supports **only Pepper, Potato, and Tomato leaf images**.

Uploading other plant species may produce incorrect predictions.

---

## 📊 Application Features

- Real-time disease prediction
- Confidence score display
- Probability distribution chart
- Disease description
- Treatment guidance
- Prevention advice
- Clean professional UI
- GPU-accelerated inference

---

## 🛠 Tech Stack

- Python 3.10+
- PyTorch
- Torchvision
- Streamlit
- Matplotlib
- Pillow
- CUDA (GPU acceleration)

---

## 🔥 Future Improvements

- Grad-CAM heatmap visualization
- Cloud deployment (Streamlit Cloud / Render)
- REST API backend (FastAPI)
- Mobile-friendly version
- Multi-crop expansion
- PDF report generation

---

## 📜 License

This project is built for educational and research purposes.

---

## 👨‍💻 Author

Built using Deep Learning, PyTorch, and GPU acceleration.
