import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch weather data from the Weather API
def fetch_weather_data(api_key, location, days=30):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt="
    weather_data = []

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = requests.get(url + date)

        if response.status_code == 200:
            data = response.json()
            daily_data = {
                "date": date,
                "sunlight_hours": data['forecast']['forecastday'][0]['day'].get('sunshine', 0),
                "cloud_cover": data['forecast']['forecastday'][0]['day'].get('cloud', 0),
                "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 0),
                "solar_energy_production": None  # Placeholder for actual production data
            }
            weather_data.append(daily_data)
        else:
            st.error(f"Error fetching data for {date}: {response.status_code}")

    return pd.DataFrame(weather_data)

# Function to create synthetic solar energy production data
def create_solar_energy_production(df):
    sunlight_factor = 1.5  # Solar energy production per sunlight hour
    temperature_factor = 0.1  # Additional energy per degree Celsius
    cloud_cover_penalty = -0.5  # Energy loss per percentage of cloud cover

    # Calculate solar energy production
    df['solar_energy_production'] = (
        df['sunlight_hours'] * sunlight_factor +
        df['temperature'] * temperature_factor +
        df['cloud_cover'] * cloud_cover_penalty
    ).clip(lower=0)  # Ensure no negative production values

    return df

# Function to plot data
def plot_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Sunlight hours plot
    axes[0].plot(df['date'], df['sunlight_hours'], marker='o', color='gold', label='Sunlight Hours')
    axes[0].set_title('Sunlight Hours Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sunlight Hours')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Cloud cover plot
    axes[1].plot(df['date'], df['cloud_cover'], marker='o', color='skyblue', label='Cloud Cover')
    axes[1].set_title('Cloud Cover Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cloud Cover (%)')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # Temperature plot
    axes[2].plot(df['date'], df['temperature'], marker='o', color='orange', label='Temperature')
    axes[2].set_title('Average Temperature Over Time')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

# Function to predict solar energy production for a new day
def predict_solar_energy(model, sunlight_hours, cloud_cover, temperature):
    input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], columns=['sunlight_hours', 'cloud_cover', 'temperature'])
    predicted_production = model.predict(input_data)
    return predicted_production[0]

# Main function to run the Streamlit app
def main():
    st.title("Solar Energy Prediction App")

    API_KEY = "15126be931c44b49917131244242510"  # Replace with your actual Weather API key
    
    LOCATION = st.text_input("Enter Location:", value="Nagpur")
    
    if st.button("Fetch Weather Data"):
        weather_df = fetch_weather_data(API_KEY, LOCATION)
        if not weather_df.empty:
            # Create synthetic solar energy production data
            weather_df = create_solar_energy_production(weather_df)
            weather_df.ffill(inplace=True)  # Forward fill missing values
            
            # Display the weather data
            st.write(weather_df)

            # Plot the data
            plot_data(weather_df)

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

            # Example prediction for a new day
            new_sunlight_hours = st.number_input("Enter Sunlight Hours for Prediction:", value=10)
            new_cloud_cover = st.number_input("Enter Cloud Cover Percentage for Prediction:", value=20)
            new_temperature = st.number_input("Enter Temperature (°C) for Prediction:", value=25)

            if st.button("Predict Solar Energy Production"):
                predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
                st.write(f'Predicted Solar Energy Production: {predicted_energy:.2f} kWh')

if __name__ == "__main__":
    main()
