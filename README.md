# PestNet 🌽

PestNet is a machine learning project developed for my Honours research in Machine Learning.  
The project focuses on automated detection of maize (corn) diseases using:

- **U-Net** for image segmentation  
- **Convolutional Neural Networks (CNNs)** for disease classification  

The system first segments affected regions on maize leaves and then classifies the disease type.

---

## 📌 Project Overview

Agricultural crop diseases significantly impact yield and food security.  
PestNet applies deep learning techniques to:

1. Segment diseased regions from maize leaf images.
2. Classify detected diseases using CNN-based models.

Although slight overfitting is observed, the models demonstrate strong convergence and stable learning behaviour.

---

## 🧠 Model Architecture

### 🔹 Segmentation
- Model: **U-Net**
- Purpose: Pixel-wise segmentation of diseased regions
- Output: Binary or multi-class segmentation masks

### 🔹 Classification
- Model: Custom CNN architecture
- Purpose: Classify maize diseases from segmented or original images
- Output: Disease category probabilities

---

## 📂 Repository Structure

```
PestNet/
│
├── classifier/        # CNN classification models and training scripts
├── segmentation/      # U-Net architecture and segmentation pipeline
├── dataset/           # Dataset loading and preprocessing scripts
├── models.py          # Model definitions and shared architectures
├── train.py           # Training pipeline
├── evaluate.py        # Evaluation scripts
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- PyTorch / TensorFlow (replace with actual framework)
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---

## 📊 Training Details

- Loss Functions:
  - Segmentation: Dice Loss / BCE (replace if needed)
  - Classification: Cross-Entropy Loss
- Optimizer: Adam
- Evaluation Metrics:
  - Accuracy
  - Precision / Recall
  - Dice Coefficient (Segmentation)

Training curves show good convergence with minor overfitting, suggesting strong learning capacity with potential for further regularization improvements.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PestNet.git
   cd PestNet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train segmentation model:
   ```bash
   python segmentation/train.py
   ```

4. Train classifier:
   ```bash
   python classifier/train.py
   ```

---

## 📈 Future Improvements

- Data augmentation to reduce overfitting
- Cross-validation
- Hyperparameter tuning
- Model pruning / regularization
- Deployment as a web or mobile application

---

## 🎓 Academic Context

This project was developed as part of my Honours research in Machine Learning, focusing on deep learning applications in agricultural disease detection.

---

## 📬 Contact

If you would like to collaborate or discuss this project, feel free to connect.

