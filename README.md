# Potato Leaf Disease Classification System

### A deep learning application for detecting potato leaf disease using transfer learning and computer vision

The system classifies potato leaf images into Healthy, Early Blight, and Late Blight categories using a MobileNetV2-based deep learning model. It provides real-time predictions, confidence scores, disease information, and interactive dashboards for model evaluation and dataset analysis.

### The modes has been trained using [Potato Leaf Disease Dataset](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset) from kaggle.

## Problem Statement

Agriculture plays a crucial role in ensuring food security and support livelihood of millions of farmers world wide. However crop disease significantly reduces agricultural productivity, resulting in economic loss and threatening food supply chains. Traditional methods of disease identification relies on manual inspection by farmers or agricultural experts, which are costly, time consuming, and is not capable to cop up with increasing food demand due to increasing population.

The objective of this project is to develop an automated potato leaf disease classification system using ML or DL techniques. The system can accurately identify the potato leaf condition from images and classify whether the plant is Healthy, Early Bright or Late Bright. Additionally, develop an interface and dashboard which can be used to identify the leaves and can show real time predictions, visualize model performance and present disease specific information, assisting users to make an informed crop management system.

## System Requirements

1. Operating System: Windows/Linux/MacOS
2. Python Version: 3.9 =< and >=3.13

## Dependencies

### Standard Python Libraries

1. `pathlib`: For file and directory management

### Third-Party Libraries

1. `tensorflow`: Deep learning model development and inference
2. `numpy`: Numerical computations
3. `pandas`: Data manipulation and analysis
4. `plotly`: Interactive visualizations
5. `streamlit`: Web application framework
6. `scikit-learn`: Model evaluation metrics
7. `joblib`: Storage of model, artifacts and metadata
8. `Pillow`: Image processing

## Features

1. Potato leaf disease classification using MobileNetV2 transfer learning.
2. Real-time image-based disease prediction.
3. Prediction confidence scoring.
4. Interactive visualization of prediction probabilities.
5. Disease information, treatment recommendations, and prevention guidelines.
6. Training performance dashboard with accuracy and loss curves.
7. Confusion matrix and classification report visualization.
8. Dataset statistics and model performance monitoring

## Model Architecture

1. Data Augmentation Layer
2. Image Preprocessing Layer
3. MobileNetV2 Feature Extraction Backbone
4. Global Average Pooling Layer
5. Dropout Layer
6. Dense Classification Layer
7. Softmax Output Layer
8. Model Retraining

## To retrain the model:

1. Open the training notebook.
2. Execute all preprocessing and training cells.
3. Train the MobileNetV2 model using the updated dataset.
4. Save the generated model and artifact files.
5. Restart the Streamlit application to use the updated model.

## Installation

Follow the steps below to set up the project locally.

### 1) Clone the repository

```bash
  git clone https://github.com/Kaush1590/Crop-Disease-Classification.git
  cd Crop-Disease-Classification
```
### 2) Create and activate a virtual environment

#### Linux/ MacOS
```bash
  python -m venv <environment_name>
  source <environment_name>/bin/activate
```

#### Windows
```bash
  python -m venv <environment_name>
  <environment_name>\Scripts\activate
```

#### Anaconda
```bash
  conda create -n <environment_name> python=3.12 -y
  conda activate <environment_name>
```

### 3) Install required dependencies
```bash
  pip install -r requirements.txt
```

### 4) Run the application

```bash
  streamlit run app.py
```

### Once started, open the URL displayed in the terminal to access the application.

## Acknowledgment
1) Kaggle for providing the Potato Leaf Disease Dataset.
2) TensorFlow and Keras development teams.
3) Open-source Python community.
