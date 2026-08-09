# Customer Churn Prediction

![Customer Churn Prediction App -Input]
(./churn1.png)

![Customer Churn Prediction App -Output]
(./churn2.png)

## Project Overview

This project predicts whether a telecom customer is likely to churn.

The project uses machine learning to analyze customer information such as:

- Contract type
- Tenure
- Monthly charges
- Internet service
- Payment method
- Customer services
- Demographic information

The final model is deployed as an interactive Streamlit application.

## Business Problem

Customer churn can cause significant revenue loss for telecom companies.

The goal of this project is to identify customers who are at higher risk of leaving so that the business can take preventive action.

## Dataset

The dataset contains customer demographic, service, contract, billing, and churn information.

Target variable:

- `Churn = Yes` → Customer churned
- `Churn = No` → Customer stayed

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.57% | 65.76% | 56.15% | 60.57% |
| Random Forest | ~78% | 63% | 48% | 54% |
| Decision Tree | 70.47% | 44.87% | 48.31% | 46.52% |

Logistic Regression was selected as the final model because it achieved the best overall performance among the tested models.

## Data Preprocessing

The project includes:

- Missing-value handling
- Categorical variable encoding
- Train/test split
- Feature scaling using StandardScaler
- Stratified sampling

## Application

The Streamlit application allows users to enter customer information and receive:

- Churn prediction
- Churn probability
- Churn risk level

### Prediction Logic

A probability threshold of 50% is used:

- Below 50% → Low Churn Risk
- 50% or above → High Churn Risk

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Chrun/
│
├── churn_analysis.ipynb
├── app.py
├── Chrun_model.pkl
├── scaler.pkl
├── feature_name.pkl
└── README.md

## How to Run
Install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit joblib
