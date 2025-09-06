# 1. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 2. LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # --- IMPORTANT: PASTE YOUR FINAL FEATURE LIST FROM THE NOTEBOOK HERE ---
    final_model_features = ['Total_Amount', 'Total_Amount_to_Repay', 'duration', 'Amount_Funded_By_Lender', 
                            'Lender_portion_Funded', 'Lender_portion_to_be_repaid', 'disbursement_month', 
                            'disbursement_day_of_week', 'disbursement_year', 'customer_Total_Amount_mean', 
                            'customer_Total_Amount_sum', 'customer_Total_Amount_std', 'customer_duration_mean', 
                            'customer_duration_max', 'customer_loan_type_count', 'country_id_Kenya', 
                            'loan_type_Type_10', 'loan_type_Type_11', 'loan_type_Type_12', 'loan_type_Type_13', 
                            'loan_type_Type_14', 'loan_type_Type_15', 'loan_type_Type_16', 'loan_type_Type_17', 
                            'loan_type_Type_18', 'loan_type_Type_19', 'loan_type_Type_2', 'loan_type_Type_20', 
                            'loan_type_Type_21', 'loan_type_Type_22', 'loan_type_Type_23', 'loan_type_Type_24', 
                            'loan_type_Type_3', 'loan_type_Type_4', 'loan_type_Type_5', 'loan_type_Type_6', 
                            'loan_type_Type_7', 'loan_type_Type_8', 'loan_type_Type_9', 'is_repeat_loan']

    # --- IMPORTANT: PASTE YOUR FULL LIST OF LOAN TYPES FROM THE NOTEBOOK HERE ---
    loan_type_options = ['Type_1', 'Type_10', 'Type_11', 'Type_12', 'Type_13', 'Type_14', 'Type_15', 'Type_16', 
                         'Type_17', 'Type_18', 'Type_19', 'Type_2', 'Type_20', 'Type_21', 'Type_22', 'Type_23', 
                         'Type_24', 'Type_3', 'Type_4', 'Type_5', 'Type_6', 'Type_7', 'Type_8', 'Type_9']

    return model, scaler, final_model_features, loan_type_options

model, scaler, final_model_features, loan_type_options = load_artifacts()

# 3. HELPER FUNCTIONS
def assign_credit_score(probability):
    if probability <= 0.003: return ('Very Low Risk', 5)
    elif 0.003 < probability <= 0.50: return ('Low Risk', 4)
    elif 0.50 < probability <= 0.90: return ('Medium Risk', 3)
    elif 0.90 < probability <= 0.98: return ('High Risk', 2)
    else: return ('Very High Risk', 1)

# 4. UI - MAIN PAGE
st.title("Africa Credit Scoring: Loan Default Prediction")
st.header("A tool to assess credit risk for new loan applications.")

# 5. UI - SIDEBAR
st.sidebar.header("Applicant Details")

def user_input_features():
    total_amount = st.sidebar.number_input("Total Loan Amount", min_value=0, value=50000)
    total_amount_to_repay = st.sidebar.number_input("Total Amount to Repay", min_value=0, value=51000)
    lender_portion_to_be_repaid = st.sidebar.number_input("Lender Portion to Repay", min_value=0, value=51000)
    duration = st.sidebar.slider("Loan Duration (days)", 1, 365, 30)
    loan_type = st.sidebar.selectbox("Loan Type", loan_type_options)
    new_vs_repeat = st.sidebar.selectbox("Customer History", ["New Loan", "Repeat Loan"])
    amount_funded = st.sidebar.number_input("Amount Funded by Lender", min_value=0, value=50000)
    lender_portion_funded = st.sidebar.slider("Lender Portion Funded (%)", 0.0, 1.0, 0.3)

    data = {
        'Total_Amount': total_amount, 'Total_Amount_to_Repay': total_amount_to_repay,
        'Lender_portion_to_be_repaid': lender_portion_to_be_repaid, 'duration': duration,
        'loan_type': loan_type, 'New_versus_Repeat': new_vs_repeat,
        'Amount_Funded_By_Lender': amount_funded, 'Lender_portion_Funded': lender_portion_funded,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 6. LOGIC & OUTPUT
if st.sidebar.button("Predict Loan Default Risk"):
    # --- Start of the Prediction Pipeline ---
    processed_df = input_df.copy()

    # Create historical/aggregation features, filling with 0 for a single prediction
    agg_features = [col for col in final_model_features if 'customer_' in col]
    for col in agg_features:
        processed_df[col] = 0

    # Create date features
    processed_df['disbursement_year'] = datetime.now().year
    processed_df['disbursement_month'] = datetime.now().month
    processed_df['disbursement_day_of_week'] = datetime.now().weekday()
            
    # One-Hot Encoding
    processed_df['is_repeat_loan'] = 1 if processed_df['New_versus_Repeat'][0] == 'Repeat Loan' else 0
    
    # OHE for loan_type. Create all possible columns and set to 0, then set the selected one to 1.
    for l_type in loan_type_options:
        col_name = f'loan_type_{l_type}'
        if col_name in final_model_features:
            processed_df[col_name] = 0 # Default to 0
    
    selected_loan_type_col = f"loan_type_{input_df['loan_type'][0]}"
    if selected_loan_type_col in final_model_features:
        processed_df[selected_loan_type_col] = 1

    processed_df['country_id_Kenya'] = 1
    
    # --- THIS IS THE KEY FIX ---
    # 1. Ensure dataframe only has the feature columns IN THE CORRECT ORDER
    processed_df_for_scaling = processed_df[final_model_features]
    
    # 2. Scale the data
    scaled_input_array = scaler.transform(processed_df_for_scaling)
    
    # 3. Convert the scaled array back to a DataFrame WITH THE CORRECT COLUMN NAMES
    scaled_input_df = pd.DataFrame(scaled_input_array, columns=final_model_features)

    # 4. Make predictions using the DataFrame
    prediction = model.predict(scaled_input_df)
    prediction_proba = model.predict_proba(scaled_input_df)
    # --- END OF FIX ---
    
    risk_category, credit_score = assign_credit_score(prediction_proba[0][1])

    # Display results
    st.subheader("Credit Risk Assessment Result")
    st.markdown("---")

    if prediction[0] == 0:
        st.success(f"**Model Prediction: LOW RISK (Will Likely Be Paid)**")
    else:
        st.error(f"**Model Prediction: HIGH RISK (Likely to Default)**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Probability of Default", value=f"{prediction_proba[0][1]:.2%}")
    with col2:
        st.metric(label="Credit Score", value=f"{credit_score} / 5")
        
    st.info(f"**Risk Category:** {risk_category}")

else:
    st.info("Please adjust the applicant details in the sidebar and click 'Predict' to see the result.")