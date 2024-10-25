import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Custom CSS for Streamlit styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 42px;
        color: #364f6b;
        font-weight: bold;
    }
    .subheader {
        font-size: 18px;
        color: #3fc1c9;
    }
    .button {
        background-color: #3fc1c9;
        color: white;
    }
    .metric {
        color: #ff9a00;
        font-size: 22px;
        font-weight: bold;
    }
    .appliance {
        font-weight: bold;
        color: #364f6b;
    }
    </style>
""", unsafe_allow_html=True)

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
                "sunlight_hours": np.random.uniform(8, 11),
                "cloud_cover": np.random.uniform(30, 80),
                "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 25),
                "solar_energy_production": None  # Placeholder for actual production data
            }
            weather_data.append(daily_data)
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

# Function to plot data
def plot_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes[0].plot(df['date'], df['sunlight_hours'], marker='o', color='gold', label='Sunlight Hours')
    axes[0].set_title('Sunlight Hours Over Time')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot(df['date'], df['cloud_cover'], marker='o', color='skyblue', label='Cloud Cover')
    axes[1].set_title('Cloud Cover Over Time')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].plot(df['date'], df['temperature'], marker='o', color='orange', label='Temperature')
    axes[2].set_title('Average Temperature Over Time')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

# Function to predict solar energy production for a new day
def predict_solar_energy(model, sunlight_hours, cloud_cover, temperature):
    input_data = pd.DataFrame([[sunlight_hours, cloud_cover, temperature]], columns=['sunlight_hours', 'cloud_cover', 'temperature'])
    predicted_production = model.predict(input_data)
    return predicted_production[0]

# Function to estimate appliance usage based on solar energy production
def estimate_appliance_usage(solar_energy):
    appliances = {
        "LED Light (10W)": solar_energy / 0.01,
        "Fan (50W)": solar_energy / 0.05,
        "Washing Machine (500W)": solar_energy / 0.5,
        "Refrigerator (150W)": solar_energy / 0.15,
        "Air Conditioner (1500W)": solar_energy / 1.5
    }
    
    st.write("### Estimated Appliance Usage (in hours)")
    for appliance, hours in appliances.items():
        st.write(f"<span class='appliance'>{appliance}</span>: {hours:.2f} hours", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.markdown("<div class='title'>Solar Energy Prediction App</div>", unsafe_allow_html=True)

    API_KEY = "15126be931c44b49917131244242510"
    LOCATION = st.text_input("Enter Location:", value="Nagpur")
    
    if st.button("Fetch Weather Data"):
        weather_df = fetch_weather_data(API_KEY, LOCATION)
        if not weather_df.empty:
            weather_df = create_solar_energy_production(weather_df)
            weather_df.ffill(inplace=True)
            
            st.write(weather_df)
            plot_data(weather_df)

            X = weather_df[['sunlight_hours', 'cloud_cover', 'temperature']]
            y = weather_df['solar_energy_production']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"<span class='metric'>Mean Squared Error: {mse:.2f}</span>", unsafe_allow_html=True)
            st.write(f"<span class='metric'>R^2 Score: {r2:.2f}</span>", unsafe_allow_html=True)

            new_sunlight_hours = st.slider("Sunlight Hours for Prediction:", 8.0, 11.0, 10.0)
            new_cloud_cover = st.slider("Cloud Cover Percentage for Prediction:", 30.0, 80.0, 50.0)
            new_temperature = st.slider("Temperature (°C) for Prediction:", 15, 45, 25)

            if st.button("Predict Solar Energy Production"):
                predicted_energy = predict_solar_energy(model, new_sunlight_hours, new_cloud_cover, new_temperature)
                st.write(f"<span class='metric'>Predicted Solar Energy Production: {predicted_energy:.2f} kWh</span>", unsafe_allow_html=True)
                estimate_appliance_usage(predicted_energy)

if __name__ == "__main__":
    main()
