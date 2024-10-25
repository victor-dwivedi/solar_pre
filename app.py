# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# API key
API_KEY = "d224a9f66ffa425eab3180904242310"

# Fetch weather data
def fetch_weather_data(api_key, location, days=30):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt="
    weather_data = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = requests.get(url + date)
        if response.status_code == 200:
            data = response.json()
            try:
                daily_data = {
                    "date": date,
                    "sunlight_hours": data['forecast']['forecastday'][0]['day'].get('sunshine', 0),
                    "cloud_cover": data['forecast']['forecastday'][0]['day'].get('cloud', 0),
                    "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 0)
                }
                weather_data.append(daily_data)
            except KeyError:
                st.error(f"Data not available for {date}")
        else:
            st.error(f"Error: {response.status_code}")
    return pd.DataFrame(weather_data)

# Synthetic solar energy production
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

# Time-of-Use (TOU) Tariff Data
def generate_tou_tariff():
    tou_hours = {
        "High Tariff": [8, 9, 10, 11, 12, 17, 18, 19],
        "Medium Tariff": [13, 14, 15, 16],
        "Low Tariff": [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23]
    }
    return tou_hours

# Appliance suggestions based on energy prediction
def suggest_appliances(predicted_energy):
    appliances = {
        "LED Bulbs": 10,
        "Laptop": 50,
        "Fan": 75,
        "Television": 150,
        "Refrigerator": 200,
        "Washing Machine": 500,
        "Air Conditioner": 2000
    }
    suggestions = [appliance for appliance, wattage in appliances.items() if predicted_energy >= wattage]
    return suggestions

# Main function
def main():
    # Background image and CSS styling
    st.markdown("""
    <style>
    .reportview-container {
        background: url('https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8c29sYXIlMjBlbmVyZ3l8ZW58MHx8MHx8fDA%3D');
        background-size: cover;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Solar Energy Production Predictor")
    st.write("Enter your location to fetch historical weather data:")

    location = st.text_input("Location", "")
    
    if location:
        weather_df = fetch_weather_data(API_KEY, location)
        if not weather_df.empty:
            weather_df = create_solar_energy_production(weather_df)

            # Train Random Forest model
            X = weather_df[['sunlight_hours', 'cloud_cover', 'temperature']]
            y = weather_df['solar_energy_production']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Generate today's solar energy production
            today_sunlight = weather_df.iloc[0]['sunlight_hours']
            today_cloud_cover = weather_df.iloc[0]['cloud_cover']
            today_temp = weather_df.iloc[0]['temperature']
            predicted_energy = model.predict([[today_sunlight, today_cloud_cover, today_temp]])[0]
            st.write(f"Predicted Solar Energy Production Today: {predicted_energy:.2f} kWh")

            # TOU Tariff Suggestion
            tou_tariff = generate_tou_tariff()
            high_tariff_hours = tou_tariff["High Tariff"]
            st.write(f"High Tariff Hours: {', '.join(map(str, high_tariff_hours))}")
            st.write("Use solar energy in high-tariff hours to save costs.")

            # Plot prediction trend
            plt.figure(figsize=(10, 5))
            plt.plot(weather_df['date'], weather_df['solar_energy_production'], marker='o', label='Production')
            plt.xticks(rotation=45)
            plt.xlabel("Date")
            plt.ylabel("Solar Energy Production (kWh)")
            plt.title("Solar Energy Production Over Last 30 Days")
            plt.legend()
            st.pyplot(plt)

            # Suggest appliances
            suggestions = suggest_appliances(predicted_energy)
            if suggestions:
                st.write("You can power the following appliances with today's solar energy:")
                st.write(", ".join(suggestions))
            else:
                st.write("Not enough solar energy for suggested appliances.")
        else:
            st.warning("Data not available for the specified location.")
            
if __name__ == "__main__":
    main()
