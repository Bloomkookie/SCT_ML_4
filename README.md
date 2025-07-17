# ✋🤖 Hand Gesture Recognition — Task 4 [CNN + LSTM + OpenCV + Transfer Learning]

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange?logo=opencv)
![Keras](https://img.shields.io/badge/DeepLearning-Keras-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)
![Model](https://img.shields.io/badge/Model-CNN%20%2B%20LSTM%20%2B%20MobileNet-blueviolet)

> ✅ **Task 4 of Internship Project** — Real-Time Hand Gesture Recognition System  
> 🎯 **Objective**: Build a hybrid ML/DL pipeline that recognizes static and dynamic hand gestures using **CNN**, **LSTM**, **Transfer Learning**, and deploys real-time detection with **OpenCV**.

---

## 📸 Demo

<img src="https://github.com/your-username/hand-gesture-recognition/blob/main/demo.gif" width="700"/>

> _Control your media player using your **hand gestures**.  
> The system shows **live predictions** with gesture class + confidence in real time._

---

## 🚀 Features

- 🖼️ **Static Gesture Classification** via CNN  
- 🎞️ **Dynamic Gestures from Video** with CNN + LSTM  
- 🔁 **Real-Time Webcam Detection** with OpenCV  
- 🧠 **Transfer Learning** with MobileNet / EfficientNet  
- 🎮 **Gesture-Based Media Controller** (Play, Pause, Volume, etc.)  
- 📊 Confusion Matrix & Performance Metrics  

---

## 📁 Dataset

- 📦 `gesture_dataset.zip` (~813MB)  
- Contains folders per gesture: `fist/`, `peace/`, `thumbs_up/`, etc.  
- Upload and unzip inside Jupyter Notebook environment.

---

## 🛠 Tech Stack

- `Python 3.8+`  
- `Jupyter Notebook`  
- `TensorFlow / Keras`  
- `OpenCV`  
- `MobileNetV2 / EfficientNet`  
- `Matplotlib`, `NumPy`, `scikit-learn`, `Pandas`

---

## 🧪 Models Used

### 1️⃣ CNN (Static Image Classifier)
- Input: 224×224 RGB
- Layers: Conv → MaxPool → Dropout → Dense → Softmax

### 2️⃣ CNN + LSTM (Video Sequence Classifier)
- CNN for per-frame feature extraction
- LSTM for temporal modeling of gesture sequence

### 3️⃣ MobileNetV2 / EfficientNet
- Pre-trained on ImageNet  
- Fine-tuned for your gesture classes

---

## 🖥️ Run Locally

### 📦 Step 1: Upload Dataset in Jupyter
- Upload your `gesture_dataset.zip`
- Unzip it in your working directory

### 📋 Step 2: Install Dependencies
```bash
pip install opencv-python tensorflow scikit-learn matplotlib
