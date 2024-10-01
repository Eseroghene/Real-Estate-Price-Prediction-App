
import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Loading the model into a variable
model = joblib.load("estate")

#Loading the Encoder into a variable
encoder = joblib.load("Binary Encoder")

# Defining function to predict prices
def predict_price(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app layout
st.title("REAL ESTATE PRICE PREDICTION APP")
st.sidebar.title("DATA SCIENCE CAPSTONE PROJECT")
st.sidebar.subheader("GROUP C")


st.header("Select Property Characteristics:")

# Input fields for the categorical variables
property_type = st.selectbox("Property",["Single Family","Two Family","Three Family","Four Family"])
residential_type = st.selectbox("Residential",["Detached House","Duplex","Triplex","Fourplex"])
face = st.selectbox("Face", ["North", "East", "South","West"])
locality = st.selectbox("Locality", ["Bridgeport", "Waterbury", "Fairfield", "West Hartford", "Greenwich", "Stamford", "Norwalk"])


# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Property': [property_type],
    'Residential': [residential_type],
    'Face': [face],
    'Locality': [locality]

})

# Perform Binary Encoding on the input data using the saved encoder
encoded_input = encoder.transform(input_data)


# Predict the price when the user clicks the button
if st.button("Predict Price"):
    input_features = np.array(encoded_input.iloc[0])  # Convert to array
    price = predict_price(input_features)
    st.write(f"The predicted price for the property is: ${price:,.2f}")

