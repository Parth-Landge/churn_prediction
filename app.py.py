import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    # Make sure this matches the exact name of your saved model file!
    with open('model.sav', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- 2. App Header & Description ---
st.set_page_config(page_title="Telco Churn Predictor", page_icon="churn", layout="centered")
st.title("Telco Customer Churn Predictor")
st.markdown("Enter customer details below to predict their churn risk. The model analyzes **33 encoded features** behind the scenes.")
st.divider()

# --- 3. User Input Layout (Using Tabs for clean UI) ---
tab1, tab2, tab3 = st.tabs(["👤 Demographics & Location", "🔌 Services Subscribed", "💳 Billing & Charges"])

with tab1:
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    
    st.subheader("Location")
    col3, col4 = st.columns(2)
    with col3:
        latitude = st.number_input("Latitude", value=34.0)
    with col4:
        longitude = st.number_input("Longitude", value=-118.0)

with tab2:
    st.subheader("Core Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    # If no phone service, force multiple lines to "No phone service"
    multi_line_options = ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"]
    multiple_lines = st.selectbox("Multiple Lines", multi_line_options)
    
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    st.subheader("Add-on Services")
    # If no internet, force add-ons to "No internet service"
    addon_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    
    col1, col2 = st.columns(2)
    with col1:
        online_security = st.selectbox("Online Security", addon_options)
        device_protection = st.selectbox("Device Protection", addon_options)
        streaming_tv = st.selectbox("Streaming TV", addon_options)
    with col2:
        online_backup = st.selectbox("Online Backup", addon_options)
        tech_support = st.selectbox("Tech Support", addon_options)
        streaming_movies = st.selectbox("Streaming Movies", addon_options)

with tab3:
    st.subheader("Contract & Billing")
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing?", ["No", "Yes"])
    with col2:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    st.subheader("Customer Value & Charges")
    col3, col4 = st.columns(2)
    with col3:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    with col4:
        cltv = st.number_input("Customer Lifetime Value (CLTV)", min_value=0, max_value=10000, value=4000)
        tenure_months = st.slider("Tenure (Months)", 0, 72, 12)
        
        # Logic to calculate your custom tenure_group_encoded (0 to 5)
        if tenure_months <= 12: tenure_group = 0
        elif tenure_months <= 24: tenure_group = 1
        elif tenure_months <= 36: tenure_group = 2
        elif tenure_months <= 48: tenure_group = 3
        elif tenure_months <= 60: tenure_group = 4
        else: tenure_group = 5

# --- 4. Data Processing & Prediction ---
st.divider()
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    # Mapping the UI inputs to the EXACT 33 columns from your df_encoded.info()
    input_data = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges,
        'CLTV': cltv,
        'Gender_Male': True if gender == "Male" else False,
        'Senior Citizen_Yes': True if senior == "Yes" else False,
        'Partner_Yes': True if partner == "Yes" else False,
        'Dependents_Yes': True if dependents == "Yes" else False,
        'Phone Service_Yes': True if phone_service == "Yes" else False,
        'Multiple Lines_No phone service': True if multiple_lines == "No phone service" else False,
        'Multiple Lines_Yes': True if multiple_lines == "Yes" else False,
        'Internet Service_Fiber optic': True if internet_service == "Fiber optic" else False,
        'Internet Service_No': True if internet_service == "No" else False,
        'Online Security_No internet service': True if online_security == "No internet service" else False,
        'Online Security_Yes': True if online_security == "Yes" else False,
        'Online Backup_No internet service': True if online_backup == "No internet service" else False,
        'Online Backup_Yes': True if online_backup == "Yes" else False,
        'Device Protection_No internet service': True if device_protection == "No internet service" else False,
        'Device Protection_Yes': True if device_protection == "Yes" else False,
        'Tech Support_No internet service': True if tech_support == "No internet service" else False,
        'Tech Support_Yes': True if tech_support == "Yes" else False,
        'Streaming TV_No internet service': True if streaming_tv == "No internet service" else False,
        'Streaming TV_Yes': True if streaming_tv == "Yes" else False,
        'Streaming Movies_No internet service': True if streaming_movies == "No internet service" else False,
        'Streaming Movies_Yes': True if streaming_movies == "Yes" else False,
        'Contract_One year': True if contract == "One year" else False,
        'Contract_Two year': True if contract == "Two year" else False,
        'Paperless Billing_Yes': True if paperless == "Yes" else False,
        'Payment Method_Credit card (automatic)': True if payment_method == "Credit card (automatic)" else False,
        'Payment Method_Electronic check': True if payment_method == "Electronic check" else False,
        'Payment Method_Mailed check': True if payment_method == "Mailed check" else False,
        'tenure_group_encoded': tenure_group
    }

    # Convert dictionary to a DataFrame for the model
    input_df = pd.DataFrame([input_data])

    # Ensure the columns are in the EXACT order as df_encoded
    # If the model complains about feature names, this guarantees exact alignment
    expected_cols = [
        'Latitude', 'Longitude', 'Monthly Charges', 'Total Charges', 'CLTV',
        'Gender_Male', 'Senior Citizen_Yes', 'Partner_Yes', 'Dependents_Yes',
        'Phone Service_Yes', 'Multiple Lines_No phone service', 'Multiple Lines_Yes',
        'Internet Service_Fiber optic', 'Internet Service_No',
        'Online Security_No internet service', 'Online Security_Yes',
        'Online Backup_No internet service', 'Online Backup_Yes',
        'Device Protection_No internet service', 'Device Protection_Yes',
        'Tech Support_No internet service', 'Tech Support_Yes',
        'Streaming TV_No internet service', 'Streaming TV_Yes',
        'Streaming Movies_No internet service', 'Streaming Movies_Yes',
        'Contract_One year', 'Contract_Two year', 'Paperless Billing_Yes',
        'Payment Method_Credit card (automatic)', 'Payment Method_Electronic check',
        'Payment Method_Mailed check', 'tenure_group_encoded'
    ]
    input_df = input_df[expected_cols]

    # Make Prediction
    try:
        prediction = model.predict(input_df)
        
        # Display Results
        if prediction[0] == 1:
            st.error("🚨 **High Risk:** This customer is likely to CHURN.")
        else:
            st.success("✅ **Low Risk:** This customer is likely to STAY.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

