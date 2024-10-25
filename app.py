# -*- coding: utf-8 -*-

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define your functions here (fetch_weather_data, create_solar_energy_production, etc.)
# Include the existing functions you have provided.

def main():
    st.title("Solar Energy Production Predictor")

    # User inputs
    api_key = st.text_input("Enter your Weather API Key", "d224a9f66ffa425eab3180904242310")
    location = st.text_input("Enter Location", "Nagpur")
    new_sunlight_hours = st.number_input("Sunlight Hours", value=10)
    new_cloud_cover = st.number_input("Cloud Cover (%)", value=20)
    new_temperature = st.number_input("Temperature (°C)", value=25)

    if st.button("Fetch Weather Data"):
        weather_df = fetch_weather_data(api_key, location)
        weather_df = create_solar_energy_production(weather_df)
        st.write(weather_df)  # Display the weather data

        # Prepare data for training
        X = weather_df[['sunlight_hours', 'cloud_cover', 'temperature']]
        y = weather_df['solar_energy_production']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success(f'Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}')

        # Example prediction
        predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
        st.success(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')

if __name__ == "__main__":
    main()
