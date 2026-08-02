import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Tourism Package Prediction App", layout="centered")

# Title and description
st.title("✈️ Tourism Package Prediction App")
st.write("This app predicts whether a customer will purchase the newly introduced Wellness Tourism Package based on their details and interactions.")

# Dynamically locate the model file (ensures it works locally and on Streamlit Cloud)
model_path = os.path.join(os.path.dirname(__file__), 'best_model.joblib')

try:
    # Load the trained pipeline
    model = joblib.load(model_path)
    model_loaded = True
except Exception as e:
    st.error(f"Model not found. Please ensure the pipeline has run and saved the model. Error: {e}")
    model_loaded = False

st.header("Customer Details & Interaction Input")

# Create layout columns
col1, col2 = st.columns(2)

# Collect inputs mapping exactly to the dataset columns (excluding CustomerID and ProdTaken)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips Annually", min_value=1, max_value=10, value=2)

with col2:
    passport = st.selectbox("Holds Passport (1: Yes, 0: No)", [0, 1])
    owncar = st.selectbox("Owns Car (1: Yes, 0: No)", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=150000, value=20000)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15)

# Prediction button
if st.button("Predict Purchase Likelihood") and model_loaded:
    # Save inputs into a dataframe with exact column names matching the training data
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeofcontact],
        'CityTier': [citytier],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_person],
        'PreferredPropertyStar': [preferred_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [passport],
        'OwnCar': [owncar],
        'NumberOfChildrenVisiting': [num_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_satisfaction],
        'ProductPitched': [product_pitched],
        'NumberOfFollowups': [num_followups],
        'DurationOfPitch': [duration_pitch]
    })
    
    # Predict using the loaded pipeline
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("🎯 The customer is **LIKELY** to purchase the Wellness Tourism Package! Target them with a campaign.")
    else:
        st.warning("🛑 The customer is **UNLIKELY** to purchase the package.")
