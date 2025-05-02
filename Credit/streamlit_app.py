import streamlit as st
import numpy as np
import pickle
import os
import traceback

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Load all models
models = {}
model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'SVM': 'svm.pkl',
}

# Load models with error handling
for model_name, file_name in model_files.items():
    try:
        if os.path.exists(file_name):
            models[model_name] = pickle.load(open(file_name, 'rb'))
            st.sidebar.success(f"Loaded {model_name} successfully")
        else:
            st.sidebar.warning(f"Warning: Model file {file_name} not found")
    except Exception as e:
        st.sidebar.error(f"Error loading model {model_name}: {e}")

# Load scaler
scaler = None
try:
    if os.path.exists('scaler.pkl'):
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        st.sidebar.success("Scaler loaded successfully")
    else:
        st.sidebar.warning("Warning: scaler.pkl not found")
except Exception as e:
    st.sidebar.error(f"Error loading scaler: {e}")

# Main page
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details and select a model to predict if the transaction is fraudulent.")

# Create form for user input
with st.form("prediction_form"):
    # Model selection
    model_name = st.selectbox("Select Model", list(models.keys()), key="model")
    
    # Input fields
    time_val = st.number_input("Time (seconds since first transaction)", key="Time")
    amount_val = st.number_input("Amount ($)", key="Amount")
    
    # Submit button
    submit_button = st.form_submit_button("Predict")

# Handle prediction
if submit_button:
    if model_name not in models:
        st.error(f"Model '{model_name}' is not available or not loaded.")
    else:
        try:
            # Scale the inputs
            if scaler:
                scaled_input = scaler.transform([[time_val, amount_val]])
            else:
                scaled_input = [[time_val, amount_val]]
            
            input_array = np.array(scaled_input).reshape(1, -1)
            
            # Make prediction
            selected_model = models[model_name]
            prediction = selected_model.predict(input_array)[0]
            
            probability = None
            if hasattr(selected_model, "predict_proba"):
                probability = selected_model.predict_proba(input_array)[0][1]
                # Only show one percentage sign
                probability_pct = f"{probability * 100:.2f}%"  # formatting to show one percentage sign
            else:
                probability_pct = "N/A"
            
            # Display result
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Transaction is Not Fraudulent")
            
            # Display additional details
            st.write(f"**Model used:** {model_name}")
            st.write(f"**Fraud probability:** {probability_pct}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error(traceback.format_exc())

# Add info in the sidebar
with st.sidebar:
    st.subheader("About")
    st.write("This application uses machine learning models to detect potential credit card fraud based on transaction time and amount.")