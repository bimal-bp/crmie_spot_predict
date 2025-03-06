import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static, st_folium
from geopy.distance import geodesic
import pickle
import math
import os
import numpy as np
from sklearn.cluster import DBSCAN

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-alert {
        color: green;
        font-weight: bold;
    }
    .warning-alert {
        color: orange;
        font-weight: bold;
    }
    .danger-alert {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load City Data
df = pd.read_pickle("cities.pkl")

# Load the Model
model_filename = "train_model.pkl"
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}. Please ensure the file exists in the same directory.")
    st.stop()

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the trained model
model = pickle.load(open('model (5).pkl', 'rb'))

# City and crime type mappings
city_names = {
    '0': 'Ahmedabad', '1': 'Bengaluru', '2': 'Chennai', '3': 'Coimbatore', '4': 'Delhi',
    '5': 'Ghaziabad', '6': 'Hyderabad', '7': 'Indore', '8': 'Jaipur', '9': 'Kanpur',
    '10': 'Kochi', '11': 'Kolkata', '12': 'Kozhikode', '13': 'Lucknow', '14': 'Mumbai',
    '15': 'Nagpur', '16': 'Patna', '17': 'Pune', '18': 'Surat'
}

crimes_names = {
    '0': 'Crime Committed by Juveniles', '1': 'Crime against SC', '2': 'Crime against ST',
    '3': 'Crime against Senior Citizen', '4': 'Crime against Children', '5': 'Crime against Women',
    '6': 'Cyber Crimes', '7': 'Economic Offences', '8': 'Kidnapping', '9': 'Murder'
}

population = {
    '0': 63.50, '1': 85.00, '2': 87.00, '3': 21.50, '4': 163.10, '5': 23.60, '6': 77.50,
    '7': 21.70, '8': 30.70, '9': 29.20, '10': 21.20, '11': 141.10, '12': 20.30, '13': 29.00,
    '14': 184.10, '15': 25.00, '16': 20.50, '17': 50.50, '18': 45.80
}

# Crime prevention suggestions
crime_suggestions = {
    '0': "Encourage educational programs and mentorship initiatives for youth.",
    '1': "Strengthen legal protection and create awareness about rights.",
    '2': "Promote inclusivity and ensure strict legal enforcement.",
    '3': "Enhance neighborhood watch programs and personal security for elders.",
    '4': "Increase child safety measures and strengthen family awareness.",
    '5': "Promote gender equality and enforce strict laws against offenders.",
    '6': "Use strong passwords, be cautious online, and report suspicious activities.",
    '7': "Be vigilant about financial frauds, verify sources before transactions.",
    '8': "Educate children about safety, avoid sharing personal details with strangers.",
    '9': "Improve community policing and strengthen law enforcement presence."
}

# Load Crime Data
@st.cache_data
def load_crime_data():
    with open('crime_data.pkl', 'rb') as file:
        return pickle.load(file)

crime_data = load_crime_data()
crime_data['state/ut'] = crime_data['state/ut'].str.title()
crime_data['district'] = crime_data['district'].str.title()

# Load Location Data
@st.cache_data
def load_location_data():
    return pd.read_pickle('state_district_lat_long.pkl')

location_data = load_location_data()
location_data['State'] = location_data['State'].str.title()
location_data['District'] = location_data['District'].str.title()

# Crime Severity Score Calculation
crime_weights = {
    'murder': 5,
    'rape': 4,
    'kidnapping & abduction': 4,
    'robbery': 3,
    'burglary': 3,
    'dowry deaths': 3
}

def calculate_crime_severity(df):
    weighted_sum = sum(df[col].sum() * weight for col, weight in crime_weights.items())
    max_possible = sum(500 * weight for weight in crime_weights.values())
    crime_index = (weighted_sum / max_possible) * 100 if max_possible > 0 else 0
    return round(crime_index, 2)

# Login Page
def login_page():
    st.title("🔐 Login Page")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    married_status = st.selectbox("Married Status", ["Single", "Married", "Divorced", "Widowed"])
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    
    if st.button("Login"):
        st.session_state['logged_in'] = True
        st.session_state['user_info'] = {
            'name': name,
            'age': age,
            'married_status': married_status,
            'gender': gender
        }
        st.success("Logged in successfully!")
        st.rerun()

# City-wise Crime Analysis
def city_wise_analysis():
    st.title("🏙 City-wise Crime Analysis")
    
    # User inputs
    city_code = st.selectbox("🏙 Select City", options=list(city_names.keys()), format_func=lambda x: city_names[x])
    crime_code = st.selectbox("⚖ Select Crime Type", options=list(crimes_names.keys()), format_func=lambda x: crimes_names[x])
    year = st.number_input("📅 Enter Year", min_value=2024, max_value=2050, step=1, value=2024)

    if st.button("🔮 Predict Crime Rate"):
        # Fetch population data for the selected city
        pop = population.get(city_code, 0)
        
        # Adjust population based on the year (assuming 1% annual growth)
        year_diff = year - 2017
        pop = pop + 0.01 * year_diff * pop
        
        try:
            # Predict crime rate using the model
            crime_rate = model.predict([[int(year), int(city_code), pop, int(crime_code)]])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Calculate estimated number of cases
        cases = math.ceil(crime_rate * pop)
        
        # Determine crime severity status
        if crime_rate <= 55:
            crime_status = "🟢 Very Low Crime Area"
            color = "green"
        elif crime_rate <= 195:
            crime_status = "🟡 Low Crime Area"
            color = "yellow"
        elif crime_rate <= 278:
            crime_status = "🟠 High Crime Area"
            color = "orange"
        else:
            crime_status = "🔴 Very High Crime Area"
            color = "red"

        # Display results with styling
        st.subheader("📊 Prediction Results")
        st.write(f"🏙 **City:** {city_names[city_code]}")
        st.write(f"⚖ **Crime Type:** {crimes_names[crime_code]}")
        st.write(f"📅 **Year:** {year}")
        st.write(f"👥 **Population:** {pop:.2f} Lakhs")
        st.markdown(f"<h3 style='color:{color};'>🚔 Predicted Cases: {cases}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{color};'>⚠ Crime Severity: {crime_status}</h3>", unsafe_allow_html=True)

        # Display crime prevention suggestion
        st.markdown("### 💡 Safety Tip:")
        st.write(f"🛑 {crime_suggestions[crime_code]}")
        
# District-wise Crime Analysis
def district_wise_analysis():
    st.title("🌍 District-wise Crime Analysis")
    state = st.selectbox('Select a State/UT:', crime_data['state/ut'].unique())

    if state:
        # Filter data for the selected state
        state_data = crime_data[crime_data['state/ut'] == state]
        
        # Compute crime severity for each district
        district_severity = {}
        trend_data = {}  # To store crime severity trends for each district

        for district in state_data['district'].unique():
            district_data = state_data[state_data['district'] == district]
            
            # Calculate crime severity for 2024
            district_severity[district] = calculate_crime_severity(district_data[district_data['year'] == 2024])
            
            # Calculate crime severity for 2022, 2023, and 2024 (trend data)
            trend_data[district] = {
                year: calculate_crime_severity(district_data[district_data['year'] == year])
                for year in [2023, 2024]
            }
        
        # Display Crime Severity Map
        st.subheader(f'Crime Severity Index for Districts in {state}')
        
        state_location = location_data[location_data['State'] == state]
        if not state_location.empty:
            latitude, longitude = state_location.iloc[0]['Latitude'], state_location.iloc[0]['Longitude']
            m = folium.Map(location=[latitude, longitude], zoom_start=7)

            for district, severity in district_severity.items():
                district_row = location_data[(location_data['State'] == state) & (location_data['District'] == district)]
                if not district_row.empty:
                    lat, lon = district_row.iloc[0]['Latitude'], district_row.iloc[0]['Longitude']
                    color = 'green' if severity < 15 else 'orange' if severity < 25 else 'red'
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"{district}: {severity}"
                    ).add_to(m)
            
            folium_static(m)
        else:
            st.warning("Coordinates for the selected state were not found.")
        
        # Crime Severity Table
        st.subheader("Crime Severity Index by District")
        df_severity = pd.DataFrame(district_severity.items(), columns=['District', 'Crime Severity Index']).sort_values(by='Crime Severity Index', ascending=False)
        st.dataframe(df_severity)

        # Recommendations for selected district
        selected_district = st.selectbox("Select a District for Detailed Analysis:", list(district_severity.keys()))
        crime_severity_index = district_severity[selected_district]
        st.metric(label="Crime Severity Index (Higher is riskier)", value=crime_severity_index)
        
        # Display Crime Severity Trend
        st.subheader("Crime Severity Trend (2022 - 2024)")
        trend_df = pd.DataFrame(trend_data[selected_district], index=["Crime Severity Index"]).T
        st.line_chart(trend_df)
        
        if crime_severity_index < 10:
            st.markdown("<div class='success-alert'>🟢 This area is relatively safe.</div>", unsafe_allow_html=True)
        elif 11<= crime_severity_index <= 25:
            st.markdown("<div class='warning-alert'>🟠 Moderate risk; stay cautious.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='danger-alert'>🔴 High risk! Precaution is advised.</div>", unsafe_allow_html=True)
# Location-wise Crime Analysis
import streamlit as st
import pandas as pd
import pickle
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static

def location_wise_analysis():
    st.title("📍 Andhra Pradesh Crime Hotspots: Check Safety of a Location")

    # Load crime location dataset
    with open('crime_data (1).pkl', 'rb') as f:
        data = pickle.load(f)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Function to check safety level within a radius
    def get_safety_level(latitude, longitude, radius_km=5):
        # Filter points within the radius
        nearby_crimes = df[
            df.apply(lambda row: geodesic((latitude, longitude), (row['Latitude'], row['Longitude'])).km <= radius_km, axis=1)
        ]
        if nearby_crimes.empty:
            return "Safe (No crimes reported in this area)", None
        # Check if any high-crime hotspots are nearby
        if (nearby_crimes['Crime Rate'] == 'High').any():
            return "High Risk", nearby_crimes
        elif (nearby_crimes['Crime Rate'] == 'Moderate').any():
            return "Moderate Risk", nearby_crimes
        else:
            return "Safe", nearby_crimes

    # Initialize session state for clicked location
    if 'clicked_location' not in st.session_state:
        st.session_state.clicked_location = None

    # Create a Folium map centered on Andhra Pradesh
    st.subheader("Interactive Map: Click to Check Safety Level")
    map_center = [16.180, 81.130]  # Center of Andhra Pradesh
    m = folium.Map(location=map_center, zoom_start=8)

    # Add a click event listener to the map
    folium.LatLngPopup().add_to(m)

    # Display the map in Streamlit
    folium_static(m)

    # Use JavaScript to capture the clicked location and send it to Streamlit
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const map = document.querySelector('.folium-map');
        map.addEventListener('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            fetch('/clicked_location?lat=' + lat + '&lng=' + lng);
        });
    });
    </script>
    """
    st.components.v1.html(js_code)

    # Check if the location is clicked and get the safety level
    if 'clicked_location' in st.session_state and st.session_state.clicked_location:
        if 'lat' in st.session_state and 'lng' in st.session_state:
            latitude = st.session_state.lat
            longitude = st.session_state.lng
            safety_level, nearby_crimes = get_safety_level(latitude, longitude)
            st.write(f"Safety Level at Latitude {latitude}, Longitude {longitude}: {safety_level}")

            # Display nearby crime data if available
            if nearby_crimes is not None:
                st.subheader("Nearby Crime Data")
                st.write(nearby_crimes[['Latitude', 'Longitude', 'Crime Rate']])

                # Display crime spots on a new map
                st.subheader("Crime Hotspots in the Area")
                m_selected = folium.Map(location=[latitude, longitude], zoom_start=12)
                for index, row in nearby_crimes.iterrows():
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=f"Crime Rate: {row['Crime Rate']}",
                        icon=folium.Icon(color='red' if row['Crime Rate'] == 'High' else 'orange')
                    ).add_to(m_selected)
                folium_static(m_selected)

# Main App Logic
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state['user_info']['name']}!")
        st.sidebar.write(f"Age: {st.session_state['user_info']['age']}")
        st.sidebar.write(f"Married Status: {st.session_state['user_info']['married_status']}")
        st.sidebar.write(f"Gender: {st.session_state['user_info']['gender']}")
        
        option = st.sidebar.radio("Choose an Analysis:", ["City-wise Crime Analysis", "District-wise Crime Analysis", "Location-wise Crime Analysis"])
        
        if option == "City-wise Crime Analysis":
            city_wise_analysis()
        elif option == "District-wise Crime Analysis":
            district_wise_analysis()
        elif option == "Location-wise Crime Analysis":
            location_wise_analysis()

if __name__ == "__main__":
    main()
