import streamlit as st
import pickle
import math
import os

# Load the model
model_filename = "train_model.pkl"  # Directly use the correct filename
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}. Please ensure the file exists in the same directory.")
    st.stop()

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define city names, crime types, and population dictionary
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

# Login Page
st.title("🔐 User Login")

name = st.text_input("👤 Name")
age = st.number_input("🎂 Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("⚥ Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("💍 Marital Status", ["Single", "Married", "Divorced", "Widowed"])

if st.button("Submit"):
    if not name:
        st.warning("Please enter your name.")
    else:
        st.success(f"Welcome, {name}!")

        # Proceed to Crime Prediction App
        st.title("🚔 Crime Rate Prediction App")
        st.write("🔍 Predict the crime rate and severity for a selected city and year.")

        # User input selection
        city_code = st.selectbox("🏙 Select City", options=list(city_names.keys()), format_func=lambda x: city_names[x])
        crime_code = st.selectbox("⚖ Select Crime Type", options=list(crimes_names.keys()), format_func=lambda x: crimes_names[x])
        year = st.number_input("📅 Enter Year", min_value=2011, max_value=2050, step=1)

        # Prediction button
        if st.button("🔮 Predict Crime Rate"):
            pop = population.get(city_code, 0)  # Ensure population exists
            year_diff = year - 2011
            pop = pop + 0.01 * year_diff * pop  # Adjusting population growth at 1% per year

            # Ensure inputs are numeric before prediction
            try:
                crime_rate = model.predict([[int(year), int(city_code), pop, int(crime_code)]])[0]
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
            st.write(f"🏙 **City:** {city_names[city_code]}")
            st.write(f"⚖ **Crime Type:** {crimes_names[crime_code]}")
            st.write(f"📅 **Year:** {year}")
            st.write(f"👥 **Population:** {pop:.2f} Lakhs")
            st.write(f"📈 **Predicted Crime Rate:** {crime_rate:.2f}")
            st.write(f"📊 **Estimated Cases:** {cases}")
            st.write(f"🚨 **Crime Severity:** {crime_status}")
