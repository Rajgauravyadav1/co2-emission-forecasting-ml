import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CO₂ Emission Forecasting - Group 6") #
st.write("Batch: 2023") #

# Load predictions
data = pd.read_csv('future_predictions.csv')

# Mitigation Report Section
st.header("Mitigation Strategies")
st.info("Focusing on proven high-emission regions identified in the analysis.")
st.markdown("""
* **Industrial Efficiency**: Deploying Carbon Capture and Storage (CCS).
* **Policy**: Implementing Carbon Pricing and trade systems.
* **Energy**: Shifting to renewable grids for top emitters.
""")

# Visualization
st.line_chart(data.set_index('year')['predicted_co2'])
