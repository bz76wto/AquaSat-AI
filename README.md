# AIS Spectrograms for MMSI Prediction using ResNet

## 🚢 Project Overview
This project leverages **Automatic Identification System (AIS) data** to generate **spectrograms** and uses **ResNet** to predict **Maritime Mobile Service Identities (MMSIs)**. By applying **deep learning to satellite and ship tracking data**, we improve **maritime risk prediction and vessel identification**.

---

## 📡 Data Processing: From AIS to Spectrograms
### **1️⃣ Extract AIS Data**
AIS provides real-time ship movement data:
- **Speed over Ground (SOG)**
- **Course over Ground (COG)**
- **Latitude/Longitude**
- **Ship Type, Destination, Heading**
- **Timestamps & Sensor Readings**

### **2️⃣ Convert AIS Data to Time-Series Signals**
Ship movement is treated as a **time-frequency signal**, processed using:
- **Short-Time Fourier Transform (STFT)**
- **Wavelet Transform (CWT)**
- **Mel-Frequency Cepstral Coefficients (MFCCs)**

### **3️⃣ Generate Spectrograms**
Spectrograms are visual representations of the ship's movement and signal patterns, created using **Matplotlib, Librosa, or Scipy**.

---

## 🧠 Model Architecture: ResNet for MMSI Classification
### **1️⃣ Load Pretrained ResNet**
We fine-tune **ResNet-18** to classify MMSIs from spectrogram images:
```python
import torch
import torchvision.models as models

resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 100)  # 100 MMSI classes
