import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# Function to calculate solar energy
def calculate_solar_energy(latitude, dT):
    # Calculate theta
    theta = 23.5 * np.sin((dT / 365.25) * 2 * np.pi)
    
    # Calculate σD
    sigma_D = 137 * np.cos(np.radians(latitude - theta))
    
    # Calculate energy over 12 hours in mJ
    E = sigma_D * (4.3e4 / np.pi)  # in mJ
    return E

# Initialize geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Streamlit UI
st.title("Solar Energy Calculation")

# User input for location
location_input = st.text_input("Enter your Location (City Name)", value="Nagpur")

# Get latitude and longitude from the location
if location_input:
    location = geolocator.geocode(location_input)
    if location:
        latitude = location.latitude
        st.write(f"Latitude for {location_input}: {latitude:.2f}")

        # User input for days since March 21
        dT = st.number_input("Days since March 21 (dT)", value=100)  # Change as needed

        # Calculate solar energy
        if st.button("Calculate Energy"):
            solar_energy = calculate_solar_energy(latitude, dT)
            st.write(f"Solar Energy over 12 hours: {solar_energy:.2f} mJ")

            # Suggest appliances based on solar energy
            if solar_energy > 1e6:  # Example threshold
                st.success("You have sufficient solar energy to power multiple appliances.")
                st.write("Suggested appliances: LED bulbs, energy-efficient fans, solar water heaters, etc.")
            else:
                st.warning("Consider using fewer appliances or reducing consumption.")

        # Plotting Solar Energy over a Year
        days_in_year = np.arange(0, 365)
        solar_energies = [calculate_solar_energy(latitude, d) for d in days_in_year]

        plt.figure(figsize=(10, 5))
        plt.plot(days_in_year, solar_energies, label='Solar Energy (mJ)', color='orange')
        plt.title('Solar Energy Collection Over a Year')
        plt.xlabel('Days Since March 21')
        plt.ylabel('Energy (mJ)')
        plt.axhline(y=np.mean(solar_energies), color='r', linestyle='--', label='Average Energy')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Additional dynamic plots
        # Seasonal Variation
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_energies = [np.mean(solar_energies[i:i + 90]) for i in [0, 90, 180, 270]]

        plt.figure(figsize=(10, 5))
        plt.bar(seasons, seasonal_energies, color='skyblue')
        plt.title('Average Solar Energy by Season')
        plt.xlabel('Season')
        plt.ylabel('Average Energy (mJ)')
        plt.grid()
        st.pyplot(plt)

        # Daily Solar Energy in a specific month (e.g., June)
        june_days = np.arange(0, 30)
        june_solar_energies = [calculate_solar_energy(latitude, d + 90) for d in june_days]  # June is about day 90 to 120

        plt.figure(figsize=(10, 5))
        plt.bar(june_days, june_solar_energies, color='lightgreen')
        plt.title('Daily Solar Energy in June')
        plt.xlabel('Day of June')
        plt.ylabel('Energy (mJ)')
        plt.grid()
        st.pyplot(plt)

        # Comparative Analysis: Energy Collection in Different Latitudes
        latitudes = [0, 30, 45, 60]
        comparative_energies = [calculate_solar_energy(lat, dT) for lat in latitudes]

        plt.figure(figsize=(10, 5))
        plt.bar(latitudes, comparative_energies, color='salmon')
        plt.title('Solar Energy at Different Latitudes')
        plt.xlabel('Latitude (degrees)')
        plt.ylabel('Energy (mJ)')
        plt.grid()
        st.pyplot(plt)

        # Energy Consumption vs. Production
        appliance_names = ['LED Bulb', 'Refrigerator', 'Washing Machine', 'Air Conditioner']
        appliance_consumption = [10, 150, 500, 1500]  # in Wh
        energy_production = [solar_energy * 1e-3 / 3600] * len(appliance_names)  # Convert mJ to Wh

        plt.figure(figsize=(10, 5))
        bar_width = 0.35
        index = np.arange(len(appliance_names))

        plt.bar(index, appliance_consumption, bar_width, label='Consumption (Wh)', color='orange')
        plt.bar(index + bar_width, energy_production, bar_width, label='Solar Energy Production (Wh)', color='lightblue')
        plt.xlabel('Appliances')
        plt.ylabel('Energy (Wh)')
        plt.title('Energy Consumption vs. Solar Energy Production')
        plt.xticks(index + bar_width / 2, appliance_names)
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    else:
        st.error("Location not found. Please try another city name.")
