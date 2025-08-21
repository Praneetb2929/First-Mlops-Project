# 🩺 Diabetes Prediction Model – Your First MLOps Project (FastAPI + Docker + K8s)

This project helps you learn Building and Deploying an ML Model using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:

✅ Model Training
✅ Building the Model locally
✅ API Deployment with FastAPI
✅ Dockerization
✅ Kubernetes Deployment
📊 Problem Statement
Predict if a person is diabetic based on:

Pregnancies
Glucose
Blood Pressure
BMI
Age
We use a Random Forest Classifier trained on the Pima Indians Diabetes Dataset.

## 🚀 Quick Start
1. Clone the Repo
```bash
git clone https://github.com/iam-veeramalla/first-mlops-project.git
cd first-mlops-project
```
2. Create Virtual Environment
```bash
python3 -m venv .mlops
source .mlops/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```

Train the Model
```bash
python train.py
```
Run the API Locally
```bash
uvicorn main:app --reload
```

Sample Input for /predict
```bash
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```
## Dockerize the API
Build the Docker Image
```bash
docker build -t diabetes-prediction-model .
```
Run the Container
```bash
docker run -p 8000:8000 diabetes-prediction-model
```
Deploy to Kubernetes
```bash
kubectl apply -f diabetes-prediction-model-deployment.yaml
```
