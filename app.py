import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💼 Salary Prediction App")
st.write("Enter details to predict salary")

st.divider()

# Input fields (ONLY 2 FEATURES)
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=65
)

# Prepare input in SAME ORDER as training
input_data = np.array([[experience, age]])

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "wb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Predict button
if st.button("🔮 Predict Salary"):
    salary = model.predict(input_data)
    st.success(f"💰 Predicted Salary: ₹ {salary[0]:,.2f}")

st.divider()
st.caption("Made with ❤️ using Streamlit")

