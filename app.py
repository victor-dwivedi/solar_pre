# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Hardcoded API key
API_KEY = "your_api_key_here"

# Function to fetch weather data from the Weather API
def fetch_weather_data(api_key, location, days=30):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt="
    weather_data = []

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = requests.get(url + date)

        if response.status_code == 200:
            data = response.json()
            try:
                # Calculate daylight hours based on sunrise and sunset times
                sunrise = data['forecast']['forecastday'][0]['astro']['sunrise']
                sunset = data['forecast']['forecastday'][0]['astro']['sunset']
                
                sunrise_dt = datetime.strptime(sunrise, '%I:%M %p')
                sunset_dt = datetime.strptime(sunset, '%I:%M %p')
                
                # Calculate daylight duration in hours
                daylight_duration = (sunset_dt - sunrise_dt).seconds / 3600

                daily_data = {
                    "date": date,
                    "sunlight_hours": daylight_duration,
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

# Function to predict solar energy production for a new day
def predict_solar_energy(model, sunlight_hours, cloud_cover, temperature):
    input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], columns=['sunlight_hours', 'cloud_cover', 'temperature'])
    predicted_production = model.predict(input_data)
    return predicted_production[0]

# Suggestions for appliances based on solar energy production, calculating usage duration
def suggest_appliances(predicted_energy):
    appliances = {
        "LED Bulbs": 10,    # watts
        "Laptop": 50,       # watts
        "Fan": 75,          # watts
        "Television": 100,  # watts
        "Refrigerator": 200, # watts
        "Washing Machine": 500, # watts
        "Air Conditioner": 2000, # watts
    }
    
    suggestions = []
    for appliance, wattage in appliances.items():
        # Convert wattage to kWh per hour and calculate hours of usage
        wattage_kwh = wattage / 1000  # converting watts to kW for kWh calculation
        usage_hours = predicted_energy / wattage_kwh
        if usage_hours > 0:
            suggestions.append((appliance, wattage, usage_hours))
    
    return suggestions

# Function to display Time of Use (ToU) tariff plot
def plot_tariff():
    hours = np.arange(24)
    tariff = [1 if (6 <= h < 18) else 2 for h in hours]  # Lower tariff in daylight, higher at night

    plt.figure(figsize=(10, 4))
    plt.plot(hours, tariff, label="ToU Tariff (1 = Low, 2 = High)", color='orange', marker='o')
    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Tariff Rate")
    plt.title("Time of Use (ToU) Tariff Throughout the Day")
    plt.legend()
    st.pyplot(plt)

# Main function
def main():
    # Set up background image and Bootstrap
    st.markdown("""
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
    .reportview-container {
        background: url('https://t3.ftcdn.net/jpg/06/58/93/12/360_F_658931267_W9QK8mbF8NvK8MrXrkek4MYE8Lr1RixM.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .content {
        background: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Solar Energy Production Predictor")
    location = st.text_input("Enter your location", placeholder="e.g., Nagpur")

    if location:
        weather_df = fetch_weather_data(API_KEY, location)
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

            # Make predictions for the entire dataset
            weather_df['predicted_energy'] = model.predict(X)

            # Display today's weather
            today_data = weather_df.iloc[0]
            st.write(f"**Today's Weather Conditions**")
            st.write(f"Sunlight Hours: {today_data['sunlight_hours']:.2f} hours")
            st.write(f"Cloud Cover: {today_data['cloud_cover']}%")
            st.write(f"Temperature: {today_data['temperature']}°C")

            # Example prediction inputs
            new_sunlight_hours = st.number_input("Sunlight hours", value=8)
            new_cloud_cover = st.number_input("Cloud cover (%)", value=20)
            new_temperature = st.number_input("Temperature (°C)", value=25)

            if st.button("Predict"):
                predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
                st.write(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')

                # Provide suggestions for appliances
                suggestions = suggest_appliances(predicted_energy)
                if suggestions:
                    st.write("### Appliance Usage Based on Predicted Solar Energy")
                    st.write("| Appliance          | Power (Watts) | Usable Hours |")
                    st.write("|--------------------|---------------|--------------|")
                    for appliance, wattage, hours in suggestions:
                        st.write(f"| {appliance}        | {wattage} W   | {hours:.2f} hrs    |")
                else:
                    st.write("Not enough energy to power any appliances.")

            # Plot ToU Tariff
            st.write("### Time of Use (ToU) Tariff")
            plot_tariff()
            st.write("**Tip:** Use high-power appliances like air conditioners and washing machines during low-tariff hours (daytime) to save on bills.")

if __name__ == "__main__":
    main()
