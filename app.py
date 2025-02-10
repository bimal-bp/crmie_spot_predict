import streamlit as st
import pandas as pd
import pickle
import math
import os
import folium
from streamlit_folium import folium_static

# Load the crime prediction model
model_filename = "train_model.pkl"
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}. Please ensure the file exists in the same directory.")
    st.stop()

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load city data from pickle file
city_file = "cities.pkl"
if not os.path.exists(city_file):
    st.error(f"City file not found: {city_file}. Please ensure the file exists in the same directory.")
    st.stop()

df = pd.read_pickle(city_file)

# Define crime categories
crimes_names = {
    0: 'Crime Committed by Juveniles', 1: 'Crime against SC', 2: 'Crime against ST',
    3: 'Crime against Senior Citizen', 4: 'Crime against children', 5: 'Crime against women',
    6: 'Cyber Crimes', 7: 'Economic Offences', 8: 'Kidnapping', 9: 'Murder'
}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Login Page
if not st.session_state.logged_in:
    st.title("🔐 User Login")

    name = st.text_input("👤 Name")
    age = st.number_input("🎂 Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("⚥ Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"])

    if st.button("Submit"):
        if not name:
            st.warning("Please enter your name.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.rerun()

# If logged in, show Crime Prediction UI
if st.session_state.logged_in:
    st.title("🚔 Crime Rate Prediction App")
    st.success(f"Welcome, {st.session_state.user_name}! 🎉")
    st.write("🔍 Predict the crime rate and severity for a selected city and year.")

    # User selects a city
    city_name = st.selectbox("🏙 Select City", df['city'].unique())
    
    # Find city latitude & longitude
    city_data = df[df['city'] == city_name]
    if not city_data.empty:
        lat, lng = city_data.iloc[0]['lat'], city_data.iloc[0]['lng']
    else:
        lat, lng = None, None

    # Display the map
    if lat is not None and lng is not None:
        city_map = folium.Map(location=[lat, lng], zoom_start=12)
        folium.Marker([lat, lng], popup=city_name, tooltip=city_name).add_to(city_map)
        st.subheader("📍 City Location")
        folium_static(city_map)

    # Select Crime Type & Year
    crime_code = st.selectbox("⚖ Select Crime Type", options=list(crimes_names.keys()), format_func=lambda x: crimes_names[x])
    year = st.number_input("📅 Enter Year", min_value=2011, max_value=2050, step=1)

    # Prediction button
    if st.button("🔮 Predict Crime Rate"):
        pop = city_data.iloc[0]['population'] if not city_data.empty else 50  # Default if no data found
        year_diff = year - 2011
        pop = pop + 0.01 * year_diff * pop  # Adjusting population growth at 1% per year

        # Predict crime rate
        try:
            crime_rate = model.predict([[int(year), city_name, pop, int(crime_code)]])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Determine crime severity
        if crime_rate <= 1:
            crime_status = "🟢 Very Low Crime Area"
        elif crime_rate <= 5:
            crime_status = "🟡 Low Crime Area"
        elif crime_rate <= 15:
            crime_status = "🔴 High Crime Area"
        else:
            crime_status = "🔥 Very High Crime Area"

        cases = math.ceil(crime_rate * pop)

        # Display results
        st.subheader("📊 Prediction Results")
        st.write(f"🏙 **City:** {city_name}")
        st.write(f"⚖ **Crime Type:** {crimes_names[crime_code]}")
        st.write(f"📅 **Year:** {year}")
        st.write(f"👥 **Population:** {pop:.2f} Lakhs")
        st.write(f"📈 **Predicted Crime Rate:** {crime_rate:.2f}")
        st.write(f"📊 **Estimated Cases:** {cases}")
        st.write(f"🚨 **Crime Severity:** {crime_status}")
