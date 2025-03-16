import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("../model/best_cloud_seeding_model.pkl")

# Mapping user inputs to numerical values
mapping = {
    "Low": {"Cloud_Type": 1, "Precipitable_Water": 20, "Cloud_Top_Temperature": -40, "Updraft_Strength": 5, "Relative_Humidity": 50, "Surface_Temperature": 5, "CAPE": 500, "Wind_Shear": 5, "Cloud_Base_Height": 500, "Dew_Point": -10},
    "Medium": {"Cloud_Type": 2, "Precipitable_Water": 40, "Cloud_Top_Temperature": -20, "Updraft_Strength": 15, "Relative_Humidity": 75, "Surface_Temperature": 15, "CAPE": 1000, "Wind_Shear": 15, "Cloud_Base_Height": 1500, "Dew_Point": 5},
    "High": {"Cloud_Type": 3, "Precipitable_Water": 60, "Cloud_Top_Temperature": -10, "Updraft_Strength": 30, "Relative_Humidity": 90, "Surface_Temperature": 25, "CAPE": 2000, "Wind_Shear": 30, "Cloud_Base_Height": 2500, "Dew_Point": 15}
}

# Set up page configuration
st.set_page_config(page_title="Cloud Seeding Prediction", page_icon="☁️", layout="centered")

# Apply custom styles
st.markdown(
    """
    <style>
        body { background-color: white; color: black; font-family: Arial; }
        div.stButton > button:first-child { background-color: #87CEEB; color: black; border-radius: 5px; }
        .stSelectbox div[data-testid="stMarkdownContainer"] { color: black; }
        .stTitle { color: #007BFF; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Cloud Seeding Feasibility Prediction")
st.write("This tool will help you predict whether cloud seeding is feasible based on atmospheric conditions.")
st.markdown("---")

# Define questions and options
questions = [
    "What is the cloud type?",
    "How much precipitable water is present?",
    "What is the cloud top temperature?",
    "What is the updraft strength?",
    "What is the relative humidity level?",
    "What is the surface temperature?",
    "What is the CAPE value?",
    "What is the wind shear?",
    "What is the cloud base height?",
    "What is the dew point?"
]

options = [["Low", "Medium", "High"]] * 10

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = []

# Step-wise selection
if st.session_state.step < len(questions):
    answer = st.selectbox(questions[st.session_state.step], options[st.session_state.step])
    
    if st.button("Next", key=f"next_{st.session_state.step}"):
        st.session_state.answers.append(answer)
        st.session_state.step += 1
        st.rerun()

# When all questions are answered, process input and make a prediction
else:
    column_names = ["Cloud_Type", "Precipitable_Water", "Cloud_Top_Temperature", 
                    "Updraft_Strength", "Relative_Humidity", "Surface_Temperature", 
                    "CAPE", "Wind_Shear", "Cloud_Base_Height", "Dew_Point"]

    # Convert user input into a Pandas DataFrame with correct column names
    input_features = pd.DataFrame([[
        mapping[st.session_state.answers[i]][column_names[i]]
        for i in range(10)
    ]], columns=column_names)

    # Make a prediction
    prediction = model.predict(input_features)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Cloud seeding is feasible.")
    else:
        st.error("❌ Cloud seeding is not feasible.")

    # Restart the process
    if st.button("Restart"):
        st.session_state.step = 0
        st.session_state.answers = []
        st.rerun()
