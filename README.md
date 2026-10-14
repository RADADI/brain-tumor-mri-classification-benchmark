# Brain Tumor Detection and Classification using Deep Learning

## Authors
- Dr.Tauhidul Alam - Professor
- Hemish Radadiya - L20574570

## Project Overview
This project presents an end-to-end deep learning system for automated brain tumor classification from MRI images. The system classifies MRI scans into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The objective is to assist in medical diagnosis by providing fast, consistent, and reliable predictions using deep learning techniques.

---

## System Requirements

### Hardware Requirements
- CPU: Intel i5 or higher
- RAM: Minimum 8 GB (16 GB recommended)
- GPU: Optional (recommended for training)
- Storage: At least 5 GB free space

---

### Software Requirements
- Python 3.8 or higher
- Operating System: Windows, Linux, or macOS

---

## Required Libraries

Install the required dependencies using the following commands:

pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install opencv-python pillow
pip install flask

---

## Project Structure

Brain_Tumor_Project_Final/
│
├── brain_tumor_web_app/
├── braintrain.py
├── braintest.py
├── README.md

---

## Dataset

This project uses the Brain Tumor MRI dataset from Kaggle.

New Dataset Link:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset Structure:

DatasetNew/
├── Training/
├── Testing/

---

## How to Run the Project

1. Open terminal in web app project folder

2. Run the web application:
python app.py

3. Open browser and go to:
http://127.0.0.1:5000/

---

## How to Use

1. Open Home Page
2. Click Get Started
3. Upload MRI image
4. Click Predict
5. View result

---

## Training the Model

python braintrain.py

---

## Testing the Model

python braintest.py

---

## Models Used

- ResNet50
- InceptionV3
- EfficientNetB0
- DenseNet121
- VGG16
- VGG19
---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

