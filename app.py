import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Load the trained model, scaler, and expected feature columns
model = keras.models.load_model('churn_ann_model.keras')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
thresholds = joblib.load('thresholds.pkl')
churn_threshold = thresholds['churn_threshold']
clv_threshold = thresholds['clv_threshold']

st.set_page_config(page_title="Telecom Churn & CLV Predictor", layout="centered")

st.title("📊 Telecom Customer Churn & CLV Predictor")
st.write("Enter a customer's details below to predict their churn risk and estimated lifetime value.")

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])

with col2:
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))
    
    st.header("Prediction")

if st.button("Predict Churn"):

    # Step 1: Build a dictionary matching the binary Yes/No columns from training
    input_dict = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'OnlineSecurity': 1 if online_security == 'Yes' else 0,
        'OnlineBackup': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection': 1 if device_protection == 'Yes' else 0,
        'TechSupport': 1 if tech_support == 'Yes' else 0,
        'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    # Step 2: One-hot columns for Contract, InternetService, PaymentMethod
    input_dict['Contract_One year'] = 1 if contract == 'One year' else 0
    input_dict['Contract_Two year'] = 1 if contract == 'Two year' else 0
    input_dict['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
    input_dict['InternetService_No'] = 1 if internet_service == 'No' else 0
    input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0
    input_dict['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0
    input_dict['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

    # Step 3: Build a DataFrame with exact column order the model expects
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Step 4: Scale the numeric columns using the SAME scaler fitted during training
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Step 5: Predict
    churn_prob = model.predict(input_df)[0][0]
    clv = monthly_charges * 24

    # Step 6: Determine risk/value segment
    high_risk = churn_prob >= churn_threshold
    high_value = clv >= clv_threshold

    if high_risk and high_value:
        segment = "High Risk / High Value"
        segment_msg = "Top priority - valuable customer likely to churn"
    elif high_risk and not high_value:
        segment = "High Risk / Low Value"
        segment_msg = "Likely to churn, lower revenue impact"
    elif not high_risk and high_value:
        segment = "Low Risk / High Value"
        segment_msg = "Valuable and stable - nurture this customer"
    else:
        segment = "Low Risk / Low Value"
        segment_msg = "Stable, routine customer"

    # Step 7: Display results
    st.subheader("Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
    with col_b:
        st.metric("Estimated CLV (24 months)", f"${clv:,.2f}")

    if churn_prob >= 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")

    st.subheader("Customer Segment")
    st.info(f"**{segment}**\n\n{segment_msg}")