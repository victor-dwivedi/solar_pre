# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import streamlit.components.v1 as components

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

# TOU Tariff Data
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
        "Phone Charger": 5,
        "Wi-Fi Router": 6,
        "Microwave Oven": 800,
        "Electric Kettle": 1200,
        "Washing Machine": 500,
        "Air Conditioner": 2000
    }
    suggestions = [appliance for appliance, wattage in appliances.items() if predicted_energy >= wattage]
    return suggestions

# Main function
def main():
    # Inject Bootstrap
    components.html(
        """
        <link
            rel="stylesheet"
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
            integrity="sha384-DZoCP7I/1IZTAJ1G2xNw2FME8M6GPFEe4oAG5F6xtqFf5jVfi1AN7HnWw5nTpuQbP"
            crossorigin="anonymous">
        """,
        height=0,
    )

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

            # Weather Condition Card
            st.markdown("""
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">Today's Weather Condition</div>
                <div class="card-body">
                    <p><strong>Sunlight Hours:</strong> {0}</p>
                    <p><strong>Cloud Cover:</strong> {1}%</p>
                    <p><strong>Temperature:</strong> {2}°C</p>
                    <p><strong>Predicted Solar Energy Production:</strong> {3:.2f} kWh</p>
                </div>
            </div>
            """.format(today_sunlight, today_cloud_cover, today_temp, predicted_energy), unsafe_allow_html=True)

            # TOU Tariff Plot
            tou_tariff = generate_tou_tariff()
            high_tariff_hours = tou_tariff["High Tariff"]
            hours = list(range(24))
            tariffs = [
                3 if hour in tou_tariff["High Tariff"] else 
                2 if hour in tou_tariff["Medium Tariff"] else 1
                for hour in hours
            ]
            plt.figure(figsize=(10, 5))
            plt.bar(hours, tariffs, color=['green' if t == 1 else 'orange' if t == 2 else 'red' for t in tariffs])
            plt.xlabel("Hour of the Day")
            plt.ylabel("Tariff Level (1 = Low, 2 = Medium, 3 = High)")
            plt.title("Time-of-Use (TOU) Tariff Plot")
            st.pyplot(plt)

            # Appliance Suggestions
            suggestions = suggest_appliances(predicted_energy)
            if suggestions:
                st.markdown("""
                <div class="card mb-4">
                    <div class="card-header bg-success text-white">Appliance Suggestions</div>
                    <div class="card-body">
                        <p>You can power the following appliances with today's solar energy:</p>
                        <ul>{}</ul>
                    </div>
                </div>
                """.format("".join(f"<li>{appliance}</li>" for appliance in suggestions)), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card mb-4">
                    <div class="card-header bg-warning text-white">Appliance Suggestions</div>
                    <div class="card-body">
                        <p>Not enough solar energy for suggested appliances.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Suggestions for Saving on Electricity Bills
            st.markdown("""
            <div class="card mb-4">
                <div class="card-header bg-info text-white">Suggestions for Saving on Electricity Bills</div>
                <div class="card-body">
                    <p>1. <strong>Use appliances during low-tariff hours</strong> (shown in green on the TOU chart) whenever possible.</p>
                    <p>2. <strong>Schedule high-power appliances like washing machines and air conditioners</strong> during peak solar production hours to save on grid energy costs.</p>
                    <p>3. <strong>Use LED bulbs and energy-efficient appliances</strong> to maximize the benefits of solar energy production.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
