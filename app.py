import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🏡 House Price Prediction App")

st.write("This AI model predicts house prices based on input features using Machine Learning.")

st.markdown("---")


# User Inputs
area = st.number_input("Area (sqft)", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)
stories = st.number_input("Number of Stories", min_value=0)
parking = st.number_input("Parking Spaces", min_value=0)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Convert categorical values to numbers
binary_map = {'yes': 1, 'no': 0}
furnishing_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}

# Predict button
if st.button("Predict Price"):
    
    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        binary_map[mainroad],
        binary_map[guestroom],
        binary_map[basement],
        binary_map[hotwaterheating],
        binary_map[airconditioning],
        parking,
        binary_map[prefarea],
        furnishing_map[furnishing]
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result
    st.success(f"🏠 Predicted Price: ₹ {int(prediction[0]):,}")

st.markdown("---")
st.markdown("💡 Built by Adarsh Mani Tiwari")