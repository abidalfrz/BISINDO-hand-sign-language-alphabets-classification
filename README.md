# 🤟 BISINDO Sign Language Detection

The repository contains a Machine Learning-based web application designed to recognize and classify Indonesian Sign Language (BISINDO) alphabets (A-Z).

Built using a microservices architecture pattern, the project separates the inference logic (Backend via **FastAPI**) from the user interface (Frontend via **Streamlit**). It utilizes a fine-tuned **ResNet18** Deep Learning pretrained model to provide accurate real-time predictions with confidence scores.

---

## 📌 Problem Statement

Communication barriers remain a significant challenge for the Deaf and Hard-of-Hearing community in Indonesia. While **BISINDO (Bahasa Isyarat Indonesia)** is a primary mode of communication for many, the vast majority of the hearing population does not understand it. This linguistic divide creates obstacles in daily interactions, education, and social inclusion.

This project aims to:
- **Bridge the communication gap** by developing a model that capable of accurately classifying BISINDO alphabets from images.
- **Demonstrate the potential of AI** in social good by applying Computer Vision techniques to solve real-world accessibility issues in Indonesia.

---

## 🧠 Features

- **Dual Input Methods**: Support for both direct image upload and real-time webcam capture.
- **Microservices Architecture**: Decoupled backend (API) and frontend for better scalability and maintenance.
- **Robust Preprocessing**: Automatic image resizing and normalization to match model requirements.

---

## 🛠️ Tech Stack

### Frontend:

- **Language**: Python
- **Framework**: Streamlit

### Backend:

- **Language**: Python
- **Framework**: FastAPI
- **ASGI Server**: Uvicorn

### Data Science & ML:

- **Data Handling**: Pandas
- **Numerical Computing**: NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Image Processing**: OpenCV, Pillow (PIL), v2
- **Deep Learning Framework**: PyTorch (Torchvision)
- **Machine Learning Algorithms**: scikit-learn, XGBoost, LightGBM, CatBoost

---

## 📁 Project Structure

```bash
BISINDO-sign-language/
├── artifacts/                  # Trained Model & Encoders
│   ├── best_model.pth          # Fine-tuned ResNet18 weights
│   └── encoder.pkl             # Label encoder (Index to Class mapping)
│
├── data/                       # Raw image dataset (A-Z)
│
├── notebooks/                  # Experiments & Training
│   └── main.ipynb              # Training pipeline & evaluation
│
├── services/                   # Backend Logic Modules
│   ├── predictor.py            # Model loading & inference logic
│   └── preprocessor.py         # Image transformation pipeline
│
├── app.py                      # FastAPI Backend Entry Point
├── frontend.py                 # Streamlit Frontend UI
├── .gitignore
├── requirements.txt            # Python dependencies list
└── README.md
```

---

## 🔁 Machine Learning Workflow

1. **Data Collection**: The BISINDO dataset contains images of hand signs representing the Indonesian alphabet (A-Z) and collected from kaggle.
2. **Exploratory Data Analysis (EDA)**: Initial analysis to understand data distribution.
3. **Data Preprocessing**: Handled in `preprocessor.py` to resize, normalize, and augment images for modeling.
4. **Model Training** : Experiments with different architectures and machine learning models in `main.ipynb`, leading to the selection of a fine-tuned ResNet18 model.
5. **Model Evaluation**: Validate using f1-score to ensure model reliability.
6. **Deployment**: Backend API built with FastAPI and frontend with Streamlit for user interaction.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[BISINDO Hand Sign Language](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)

We would like to acknowledge and thanks to the dataset creator for making this resource publicly available for research and educational use.

## 🚀 How to Run

Download the dataset from the provided Kaggle [Link](#-dataset--credits) first and place it in the `data/` directory before running the application.

### 1. Clone the Repository:

Open your terminal and run the following commands:

```bash
git clone https://github.com/abidalfrz/BISINDO-hand-sign-language-alphabets-classification.git
cd BISINDO-hand-sign-language-alphabets-classification
```

### 2. Create a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the applications:

You need to run two separate terminals for backend and frontend.

<b>Terminal 1 - FastAPI (Backend):</b>

```bash
python app.py

# The API will be accessible at http://localhost:8000
```

<b>Terminal 2 - Streamlit (Frontend):</b>

```bash
streamlit run frontend.py
# The webapp will be accessible at http://localhost:8501
```

### 5. Access the Application

Open your web browser and the Streamlit app will automatically open at: http://localhost:8501

1. Choose between uploading an image file or using your webcam to capture a sign.
2. Click Analyze Image to see the predicted alphabet and confidence score.

---






