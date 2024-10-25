# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Function to fetch weather data from the Weather API
def fetch_weather_data(api_key, location, days=30):
    # Base URL for the weather API
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt="
    weather_data = []

    # Get the last 'days' days of weather data
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = requests.get(url + date)

        if response.status_code == 200:
            data = response.json()
            try:
                # Extract relevant information
                daily_data = {
                    "date": date,
                    "sunlight_hours": data['forecast']['forecastday'][0]['day'].get('sunshine', 0),
                    "cloud_cover": data['forecast']['forecastday'][0]['day'].get('cloud', 0),
                    "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 0),
                    "solar_energy_production": None
                }
                weather_data.append(daily_data)
            except KeyError as e:
                st.error(f"Key error: {e} on date: {date}")
                st.write("Response data:", data)
        else:
            st.error(f"Error fetching data for {date}: {response.status_code}")

    return pd.DataFrame(weather_data)

# Function to create synthetic solar energy production data
def create_solar_energy_production(df):
    sunlight_factor = 1.5
    temperature_factor = 0.1
    cloud_cover_penalty = -0.5

    df['solar_energy_production'] = (
        df['sunlight_hours'] * sunlight_factor +
        df['temperature'] * temperature_factor +
        df['cloud_cover'] * cloud_cover_penalty
    ).clip(lower=0)

    return df

# Main function
def main():
    st.title("Solar Energy Production Predictor")

    api_key = st.text_input("Enter your Weather API key", "")
    location = st.text_input("Enter your location", "Nagpur")

    if api_key and location:
        weather_df = fetch_weather_data(api_key, location)
        if weather_df is not None and not weather_df.empty:
            weather_df = create_solar_energy_production(weather_df)

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

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f'Mean Squared Error: {mse:.2f}')
            st.write(f'R^2 Score: {r2:.2f}')

            # Function to predict solar energy production for a new day
            def predict_solar_energy(model, sunlight_hours, cloud_cover, temperature):
                input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], columns=['sunlight_hours', 'cloud_cover', 'temperature'])
                predicted_production = model.predict(input_data)
                return predicted_production[0]

            # Example prediction inputs
            new_sunlight_hours = st.number_input("Sunlight hours", value=10)
            new_cloud_cover = st.number_input("Cloud cover (%)", value=20)
            new_temperature = st.number_input("Temperature (°C)", value=25)

            if st.button("Predict"):
                predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
                st.write(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')
        else:
            st.warning("No data available for the specified location and date range.")

# Run the app
if __name__ == "__main__":
    main()
