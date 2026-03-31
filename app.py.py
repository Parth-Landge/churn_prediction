### Step 3: The Streamlit Code (`app.py`)
Because you used `pd.get_dummies(drop_first=True)`, your model expects around 34 specific columns (like `Gender_Male`, `Internet Service_Fiber optic`, etc.). 

The trick to Streamlit is to ask the user simple questions, and then let Python translate their answers into the exact 1s and 0s your model expects. Here is your starter code.

```python
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Model ---
# st.cache_resource ensures the model is only loaded once, making the app faster
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- 2. App Header & Description ---
st.title("📉 Telco Customer Churn Predictor")
st.write("Enter the customer's details below to predict their likelihood of canceling their service.")

# --- 3. User Input Layout (Using Columns for a clean UI) ---
st.header("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    # Continuous Variables
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    cltv = st.number_input("Customer Lifetime Value (CLTV)", min_value=0, max_value=10000, value=4000)
    
    # Tenure Group (Converting slider directly to your Ordinal Encoded 0-5 scale)
    tenure_months = st.slider("Tenure (Months)", 0, 72, 12)
    if tenure_months <= 12: tenure_group = 0
    elif tenure_months <= 24: tenure_group = 1
    elif tenure_months <= 36: tenure_group = 2
    elif tenure_months <= 48: tenure_group = 3
    elif tenure_months <= 60: tenure_group = 4
    else: tenure_group = 5

with col2:
    # Categorical Variables
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    dependents = st.radio("Has Dependents?", ["Yes", "No"])

# --- 4. Data Processing (The Magic Step) ---
# We create a dictionary with ALL your model's exact column names set to False/0 by default.
# NOTE: You MUST update this dictionary to match your exact X_train columns!
if st.button("Predict Churn Risk", type="primary"):
    
    # 1. Initialize all features to 0 / False
    input_data = {
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges,
        'CLTV': cltv,
        'tenure_group': tenure_group,
        'Dependents_Yes': True if dependents == "Yes" else False,
        'Contract_One year': True if contract == "One year" else False,
        'Contract_Two year': True if contract == "Two year" else False,
        'Internet Service_Fiber optic': True if internet_service == "Fiber optic" else False,
        'Internet Service_No': True if internet_service == "No" else False,
        'Payment Method_Credit card (automatic)': True if payment_method == "Credit card (automatic)" else False,
        'Payment Method_Electronic check': True if payment_method == "Electronic check" else False,
        'Payment Method_Mailed check': True if payment_method == "Mailed check" else False,
        # ... ADD ALL YOUR OTHER COLUMNS HERE (Gender_Male, Senior Citizen_Yes, etc.) ...
    }

    # 2. Convert to DataFrame (so it matches what the model trained on)
    input_df = pd.DataFrame([input_data])

    # 3. Make Prediction
    try:
        prediction = model.predict(input_df)
        
        # 4. Display Results
        st.divider()
        if prediction[0] == 1:
            st.error("🚨 **High Risk:** This customer is likely to CHURN.")
        else:
            st.success("✅ **Low Risk:** This customer is likely to STAY.")
            
    except Exception as e:
        st.error(f"Error making prediction. Make sure all columns exactly match your training data! Details: {e}")
```

### Step 4: How to Run Your App
Running a Streamlit app is wonderfully simple. 

1. Open your **Terminal** or **Anaconda Prompt**.
2. Navigate to your project folder using the `cd` command (e.g., `cd path/to/your/telco_churn_app`).
3. Install your libraries (you only need to do this once):
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit server:
   ```bash
   streamlit run app.py