import streamlit as st   #convert python code into website
import pandas as pd       #it create the customer input data in the same table format our model expects
import joblib             #load model and scaler what we saved

#load saved files
model = joblib.load("Chrun_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_name.pkl")

#website title
st.title("📊 Customer Churn Prediction")
st.write("Predict Wheather a Telecom Customer is Likely to Churn"
         "based on their service,Contract and Billing information")
st.subheader("Customer Information")

#Gender
Gender = st.selectbox("gender", ["Female","Male"])

#Senior Citizen
seniorcitizen = st.selectbox("SeniorCitizen",[0,1])

#partner
partner = st.selectbox("Partner",["No","Yes"])

#department
dependents = st.selectbox("Departments",["No","Yes"])

#Tenure
tenure= st.number_input("tenure",min_value=0,
                     max_value=72, value=12)


#Phone Service
PhoneService = st.selectbox("PhoneService",["No","Yes"])

#Multiple_Lines
MultipleLines = st.selectbox("MultipleLines",["No","Yes","No phone service"])

#Internet_Service
InternetService = st.selectbox("InternetService",["DSL","Fiber optic","No"])

#Online_security
OnlineSecurity= st.selectbox("OnlineSecurity",["No","Yes","No phone service"])

#Online_Backup
OnlineBackup = st.selectbox("OnlineBackup",["No","Yes","No phone service"])

#Device_Protection
DeviceProtection = st.selectbox("DeviceProtection",["No","Yes","No phone service"])

#tech_Support
TechSupport= st.selectbox("TechSupport",["No","Yes","No phone service"])

#Streaming_tv
StreamingTV = st.selectbox("StreamingTV",["No","Yes","No phone service"])

#Streaming_movies
StreamingMovies = st.selectbox("StreamingMovies",["No","Yes","No phone service"])

#Contract
Contract = st.selectbox("Contract",["Month-to-Month","One year","Two year"])

#Paperless_Billing
PaperlessBilling = st.selectbox("PaperlessBilling",["No","Yes"])

#PaymentMethod
PaymentMethod = st.selectbox("PaymentMethod",["Electronic check","Mailed check",
                             "Bank transfer (automatic)","Credit card (automatic)"])

#Monthly_charges
MonthlyCharges= st.number_input("MonthlyCharges",min_value=0.0,value=70.0
                               )

#Total_charges
TotalCharges= st.number_input("TotalCharges",min_value=0.0,value=1000.0)



#PANDAS DATAFRAME
input_data = pd.DataFrame({
    "gender": [Gender],
    "SeniorCitizen": [seniorcitizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

input_data = pd.get_dummies(input_data,drop_first=True)
input_data = input_data.reindex(columns=feature_names,fill_value=0)
input_scaler = scaler.transform(input_data)


if st.button("Predict Chrun"):
    predict = model.predict(input_scaler)[0]
    probability = model.predict_proba(input_scaler)[0][1]
    if predict==1:
        st.error("⚠️High Churn Risk")
    else: st.success("✅Low churn Risk")
    st.write(f"Chrun Probability : {probability:2%}")


