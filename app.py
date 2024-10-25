import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

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
                daily_data = {
                    "date": date,
                    "sunlight_hours": data['forecast']['forecastday'][0]['day'].get('sunshine', 0),
                    "cloud_cover": data['forecast']['forecastday'][0]['day'].get('cloud', 0),
                    "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 0),
                    "solar_energy_production": None  # Placeholder for actual production data
                }
                weather_data.append(daily_data)
            except KeyError as e:
                print(f"Key error: {e} on date: {date}")
                print("Response data:", data)
        else:
            print(f"Error fetching data for {date}: {response.status_code}")

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

# Function to train a Random Forest model
def train_model(df):
    X = df[['sunlight_hours', 'cloud_cover', 'temperature']]
    y = df['solar_energy_production']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to create synthetic tariff data
def create_tariff_data(df):
    # Example tariff data (this can be more complex based on actual usage data)
    df['tariff'] = np.where(df['solar_energy_production'] > 0, 0.12, 0.20)  # 0.12 for solar, 0.20 for grid
    return df

# Function to predict solar energy production for a new day
def predict_solar_energy(model, sunlight_hours, cloud_cover, temperature):
    input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], columns=['sunlight_hours', 'cloud_cover', 'temperature'])
    predicted_production = model.predict(input_data)
    return predicted_production[0]

# Streamlit app
def main():
    st.title("Solar Energy Prediction App")
    
    # Replace with your actual API key and desired location
    API_KEY = "d224a9f66ffa425eab3180904242310"  # Add your Weather API key here
    LOCATION = st.text_input("Enter Location:", value="Nagpur")

    if st.button("Fetch Weather Data"):
        weather_df = fetch_weather_data(API_KEY, LOCATION)
        weather_df = create_solar_energy_production(weather_df)
        weather_df = create_tariff_data(weather_df)

        # Display weather data
        st.write("### Weather Data")
        st.dataframe(weather_df)

        # Train model
        model = train_model(weather_df)

        # Dynamic plots
        st.write("### Weather Conditions and Solar Energy Production")
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(weather_df['date'], weather_df['sunlight_hours'], marker='o')
        axs[0].set_title('Sunlight Hours')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Hours')
        
        axs[1].plot(weather_df['date'], weather_df['cloud_cover'], marker='o', color='orange')
        axs[1].set_title('Cloud Cover')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Percentage (%)')
        
        axs[2].plot(weather_df['date'], weather_df['solar_energy_production'], marker='o', color='green')
        axs[2].set_title('Solar Energy Production')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('kWh')

        st.pyplot(fig)

        # User input for prediction
        st.write("### Predict Solar Energy Production for a New Day")
        new_sunlight_hours = st.number_input("Enter Sunlight Hours:", value=10)
        new_cloud_cover = st.number_input("Enter Cloud Cover (%):", value=20)
        new_temperature = st.number_input("Enter Temperature (°C):", value=25)

        if st.button("Predict"):
            predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
            st.write(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')

        # Show synthetic tariff data
        st.write("### Synthetic Tariff Data")
        st.line_chart(weather_df['tariff'])

if __name__ == "__main__":
    main()
