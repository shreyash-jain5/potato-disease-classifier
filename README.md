# Potato Disease Classifier

This project uses a Convolutional Neural Network (CNN) to classify potato leaf diseases into:

- Early Blight
- Late Blight
- Healthy

## 🔧 Tech Stack

- TensorFlow / Keras
- FastAPI (Backend API)
- React.js (Frontend)
- Google Cloud Platform (Deployment)

## 🚀 Features

- Real-time image-based disease classification
- REST API using FastAPI
- Web interface using React.js
- Model trained with data augmentation on PlantVillage dataset

## 🧠 Model Details

A custom CNN was trained using TensorFlow on three classes of potato leaf images:
- potato_early_blight
- potato_late_blight
- potato_healthy

The model achieves good accuracy and is integrated with a FastAPI server for prediction.

## 📁 Project Structure

potato-disease-classifier/
├── training/ # Contains training scripts and dataset
│ ├── train.py # CNN model training script
│ └── model.h5 # Trained Keras model
├── api/ # FastAPI backend
│ ├── main.py # API for image classification
│ └── utils.py # Helper functions for preprocessing
├── web/ # React frontend (image upload and result display)
├── README.md # Project documentation (this file)


## 🧪 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyash-jain5/potato-disease-classifier.git
   cd potato-disease-classifier
2. Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install dependencies
   pip install -r requirements.txt

4. Run the FastAPI server
   uvicorn main:app --reload

👤 Author
Shreyash Jain
