import streamlit as st
import pandas as pd
import pickle
import math
import os

# Load City Data
df = pd.read_pickle("cities.pkl")

# Load the Model
model_filename = "train_model.pkl"
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}. Please ensure the file exists in the same directory.")
    st.stop()

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define city names, crime types, and population data
city_names = {
    0: 'Ahmedabad', 1: 'Bengaluru', 2: 'Chennai', 3: 'Coimbatore', 4: 'Delhi',
    5: 'Ghaziabad', 6: 'Hyderabad', 7: 'Indore', 8: 'Jaipur', 9: 'Kanpur',
    10: 'Kochi', 11: 'Kolkata', 12: 'Kozhikode', 13: 'Lucknow', 14: 'Mumbai',
    15: 'Nagpur', 16: 'Patna', 17: 'Pune', 18: 'Surat'
}

crimes_names = {
    0: 'Crime Committed by Juveniles', 1: 'Crime against SC', 2: 'Crime against ST',
    3: 'Crime against Senior Citizen', 4: 'Crime against children', 5: 'Crime against women',
    6: 'Cyber Crimes', 7: 'Economic Offences', 8: 'Kidnapping', 9: 'Murder'
}

population = {
    0: 63.50, 1: 85.00, 2: 87.00, 3: 21.50, 4: 163.10, 5: 23.60, 6: 77.50, 7: 21.70, 8: 30.70,
    9: 29.20, 10: 21.20, 11: 141.10, 12: 20.30, 13: 29.00, 14: 184.10, 15: 25.00, 16: 20.50,
    17: 50.50, 18: 45.80
}

# Load Crime Data
@st.cache_data
def load_crime_data():
    with open('crime_data.pkl', 'rb') as file:
        return pickle.load(file)

crime_data = load_crime_data()
crime_data['state/ut'] = crime_data['state/ut'].str.title()
crime_data['district'] = crime_data['district'].str.title()

# UI Navigation
st.title("🚔 Crime Analysis Dashboard")
option = st.radio("Choose an Analysis:", ("Crime Rate Prediction", "Overall Crime Analysis"))

if option == "Crime Rate Prediction":
    st.title("🚔 Crime Rate Prediction App")
    city_code = st.selectbox("🏙 Select City", options=list(city_names.keys()), format_func=lambda x: city_names[x])
    crime_code = st.selectbox("⚖ Select Crime Type", options=list(crimes_names.keys()), format_func=lambda x: crimes_names[x])
    year = st.number_input("📅 Enter Year", min_value=2011, max_value=2050, step=1)

    if st.button("🔮 Predict Crime Rate"):
        pop = population.get(city_code, 0)
        year_diff = year - 2011
        pop = pop + 0.01 * year_diff * pop
        
        try:
            crime_rate = model.predict([[int(year), int(city_code), pop, int(crime_code)]])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        cases = math.ceil(crime_rate * pop)
        crime_status = "🟢 Very Low Crime Area" if crime_rate <= 1 else \
                       "🟡 Low Crime Area" if crime_rate <= 5 else \
                       "🔴 High Crime Area" if crime_rate <= 15 else \
                       "🔥 Very High Crime Area"

        st.subheader("📊 Prediction Results")
        st.write(f"🏙 **City:** {city_names[city_code]}")
        st.write(f"⚖ **Crime Type:** {crimes_names[crime_code]}")
        st.write(f"📅 **Year:** {year}")
        st.write(f"👥 **Population:** {pop:.2f} Lakhs")
        st.write(f"📈 **Predicted Crime Rate:** {crime_rate:.2f}")
        st.write(f"📊 **Estimated Cases:** {cases}")
        st.write(f"🚨 **Crime Severity:** {crime_status}")

elif option == "Overall Crime Analysis":
    st.title("📊 Overall Crime Analysis")
    import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static

# Load the dataset
@st.cache_data
def load_crime_data():
    return pd.read_pickle('crime_data.pkl')

@st.cache_data
def load_location_data():
    return pd.read_pickle('state_district_lat_long.pkl')

crime_data = load_crime_data()
location_data = load_location_data()

crime_data['state/ut'] = crime_data['state/ut'].str.title()
crime_data['district'] = crime_data['district'].str.title()
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

# User Input: Select State
st.title("🌍 Crime Risk Analysis for All Districts in a State")
state = st.selectbox('Select a State/UT:', crime_data['state/ut'].unique())

if state:
    # Filter data for the selected state
    state_data = crime_data[crime_data['state/ut'] == state]
    
    # Compute crime severity for each district
    district_severity = {}
    for district in state_data['district'].unique():
        district_data = state_data[state_data['district'] == district]
        district_severity[district] = calculate_crime_severity(district_data)
    
    # Map Visualization
    st.subheader("Crime Hotspot Map for All Districts")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=6)  # Center map on India
    
    for district, severity in district_severity.items():
        location_row = location_data[(location_data['State'] == state) & (location_data['District'] == district)]
        if not location_row.empty:
            lat, lon = location_row.iloc[0]['Latitude'], location_row.iloc[0]['Longitude']
            
            # Assign colors based on severity index
            if severity < 45: 
                color = 'green'
            elif 25 <= severity <= 75: 
                color = 'orange'
            else:
                color = 'red'
            
            # Add CircleMarker to the map
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

