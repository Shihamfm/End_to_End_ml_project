# 🚀 End-to-End Machine Learning Project - Credit card fraud detection 

## 📌 Overview
This project demonstrates a complete **end-to-end machine learning pipeline**, covering the full lifecycle from data ingestion to deployment.

It follows **industry best practices (MLOps + modular architecture)** to build scalable, production-ready ML systems.

The solution shows how raw data can be transformed into **actionable predictions** through structured pipelines and automation.

---

## 🎯 Business Problem
The objective of this project is to:
- Process raw data efficiently  
- Build robust machine learning models  
- Generate reliable predictions  
- Enable real-time inference for decision-making  

This pipeline can be applied to:
- Customer analytics  
- Fraud detection  
- Demand forecasting  
- Predictive maintenance  

---

## 🧠 Key Features

- ✅ End-to-End ML Pipeline  
  (Data ingestion → preprocessing → training → evaluation → deployment)

- ✅ Modular & Scalable Architecture  

- ✅ Feature Engineering Pipeline  

- ✅ Model Training & Hyperparameter Tuning  

- ✅ Experiment Tracking  

- ✅ API-Based Deployment  

- ✅ Reproducibility  

---

## 🏗️ Tech Stack

**Languages & Libraries**
- Python  
- Pandas, NumPy  
- Scikit-learn  

**Tools & MLOps**
- Docker  
- GitHub Actions (CI/CD)  

**Concepts**
- Data Engineering  
- Machine Learning  
- Model Evaluation  
- API Deployment  

---

## ⚙️ Project Architecture

```bash
Data Source
↓
Data Ingestion
↓
Data Preprocessing & Feature Engineering
↓
Feature Store
↓
Model Training & Hyperparameter Tuning
↓
Model Evaluation
↓
Model Deployment (API)
↓
Real-time Inference
```

## 📂 Project Structure

```bash
├── notebooks          # Exploratory Data Analysis (EDA)
├── src
│   ├── config         # Configuration files
│   ├── feature_store  # Data ingestion & transformation
│   ├── training       # Model training & evaluation
│   ├── inference      # Deployment & prediction API
│   └── utils          # Logging, configs, helpers
├── tests              # Unit testing
├── Dockerfile         # Containerization
├── Makefile           # Pipeline automation
```

---

## 🔄 ML Pipeline Workflow

1. Data ingestion from source  
2. Data preprocessing and transformation  
3. Feature engineering and storage  
4. Train/test split  
5. Model training (multiple algorithms)  
6. Hyperparameter tuning  
7. Model evaluation and selection  
8. Model deployment via API  
9. Real-time predictions  

---

## 📊 Model Development

- Multiple ML models are trained and compared  
- Hyperparameter tuning is applied  
- Best-performing model is selected based on evaluation metrics  
- Final model is deployed for inference  

---

## 🚀 How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/Shihamfm/End_to_End_ml_project
cd End_to_End_ml_project
```
### 2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # Mac/Linux

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run Pipeline
make train
make evaluate

### 5. Run Inference (API)
python app.py
