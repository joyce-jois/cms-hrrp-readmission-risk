# 🏥 CMS Hospital Readmission Risk Prediction
**Explainable & Uncertainty-Aware Machine Learning with Random Forests**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare-green)

## 📌 Project Overview
This project predicts hospital readmission risk using CMS Hospital Readmissions Reduction Program (HRRP) data.
The system is built with an emphasis on **explainability, uncertainty quantification, and deployment readiness**.

## 🧠 Methods Summary
A Random Forest ensemble classifier was trained on engineered hospital-level features.
Prediction uncertainty is estimated using ensemble variance, and global explainability is provided via feature importance
and permutation-based methods.

## 🚀 Deployment
- Streamlit application
- Deployed via Streamlit Cloud

## 📊 Domain
Healthcare | Predictive Modeling | Explainable AI

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
