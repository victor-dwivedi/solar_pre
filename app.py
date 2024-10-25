# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Hardcoded API key
API_KEY = "6905b6c823214623958124021242510"

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

                # Retrieve cloud cover data
                cloud_cover = data['forecast']['forecastday'][0]['day'].get('cloud', None)
                
                # Skip if cloud cover data is not available
                if cloud_cover is None:
                    continue

                daily_data = {
                    "date": date,
                    "sunlight_hours": daylight_duration,
                    "cloud_cover": cloud_cover,
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
    input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], 
                              columns=['sunlight_hours', 'cloud_cover', 'temperature'])
    predicted_production = model.predict(input_data)
    return predicted_production[0]

# Suggestions for appliances based on solar energy production
def suggest_appliances(predicted_energy):
    appliances = {
        "LED Bulbs": 2,     # watts
        "Laptop": 50,       # watts
        "Television": 100,   # watts
        "Refrigerator": 200, # watts
        "Washing Machine": 500, # watts
        "Air Conditioner": 2000, # watts
        "Electric Kettle": 1500, # watts
        "Microwave Oven": 1000,  # watts
        "Fan": 70,         # watts
        "Desktop Computer": 200  # watts
    }
    
    suggestions = {}
    for appliance, wattage in appliances.items():
        if predicted_energy >= wattage:
            hours = predicted_energy / wattage  # Calculate usage hours
            suggestions[appliance] = hours
    
    return suggestions

# Plotting functions
def plot_energy_production(weather_df):
    plt.figure(figsize=(10, 5))
    plt.plot(weather_df['date'], weather_df['predicted_energy'], marker='o', color='orange')
    plt.fill_between(weather_df['date'], 0, weather_df['predicted_energy'], color='orange', alpha=0.3)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Solar Energy Production (kWh)")
    plt.title("Predicted Solar Energy Production Over Last 30 Days")
    plt.grid()
    st.pyplot(plt)

def plot_sunlight_hours(weather_df):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=weather_df, x='date', y='sunlight_hours', color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("Sunlight Hours")
    plt.title("Sunlight Hours Over the Last 30 Days")
    plt.grid()
    st.pyplot(plt)

def plot_temperature(weather_df):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=weather_df, x='date', y='temperature', marker='o', color='green')
    plt.fill_between(weather_df['date'], weather_df['temperature'], color='green', alpha=0.3)
    plt.xticks(rotation=45)
    plt.ylabel("Temperature (°C)")
    plt.title("Average Temperature Over the Last 30 Days")
    plt.grid()
    st.pyplot(plt)

def plot_cloud_cover(weather_df):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=weather_df, x='date', y='cloud_cover', marker='o', color='gray')
    plt.fill_between(weather_df['date'], weather_df['cloud_cover'], color='gray', alpha=0.3)
    plt.xticks(rotation=45)
    plt.ylabel("Cloud Cover (%)")
    plt.title("Cloud Cover Over the Last 30 Days")
    plt.grid()
    st.pyplot(plt)

def plot_energy_vs_weather(weather_df):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=weather_df, x='sunlight_hours', y='predicted_energy', color='purple')
    plt.xlabel("Sunlight Hours")
    plt.ylabel("Predicted Solar Energy Production (kWh)")
    plt.title("Energy Production vs. Sunlight Hours")
    plt.grid()
    st.pyplot(plt)

# Main function
def main():
    # Set up background image
    st.markdown(
    """
    <style>
    .reportview-container {
        background-image: url('https://t3.ftcdn.net/jpg/06/58/93/12/360_F_658931267_W9QK8mbF8NvK8MrXrkek4MYE8Lr1RixM.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.title("Solar Energy Production Predictor")
    
    location = st.text_input("Enter your location", "Nagpur")

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

            # Display today's date
            today_date = datetime.now().strftime("%Y-%m-%d")
            st.write(f"Today's date: {today_date}")

            # Plotting
            plot_energy_production(weather_df)
            plot_sunlight_hours(weather_df)
            plot_temperature(weather_df)
            plot_cloud_cover(weather_df)
            plot_energy_vs_weather(weather_df)

            # Input for new day prediction
            new_sunlight_hours = st.number_input("Sunlight hours", value=10)
            new_cloud_cover = st.number_input("Cloud cover (%)", value=20)
            new_temperature = st.number_input("Temperature (°C)", value=25)

            if st.button("Predict"):
                predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
                st.write(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')

                # Provide suggestions for appliances
                suggestions = suggest_appliances(predicted_energy)
                if suggestions:
                    st.write("You can power the following appliances with the predicted solar energy:")
                    st.write("Appliance - Hours of usage:")
                    usage_table = pd.DataFrame(suggestions.items(), columns=['Appliance', 'Hours of Usage'])
                    st.table(usage_table)
                else:
                    st.write("Not enough energy to power any appliances.")
        else:
            st.warning("No data available for the specified location.")

if __name__ == "__main__":
    main()
