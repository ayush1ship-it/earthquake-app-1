pip install --upgrade pip
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# Load the trained model
with open('random_forest_regressor.pkl', 'rb') as pickf:
    model = pickle.load(pickf)

# --- Set Page Config ---
st.set_page_config(page_title="Earthquake Predictor", layout="wide")
st.markdown("<style>.stApp {background-color:#f0f8ff;}</style>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; margin-top:0px; padding-top:0px;'>Earthquake Predictor</h2>",
            unsafe_allow_html=True)
st.write("<h4 style='text-align: center; margin-top:0px; padding-top:0px;'><i>Transforming Seismic Data into "
         "Life-Saving Insights</i></h4>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Enter Details")
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    inp_latitude = st.number_input("Latitude (24.0° N to 46° N)", min_value=24.0, max_value=46.0, value=35.47, step=0.000001, format="%0.6f")
    inp_longitude = st.number_input("Longitude (122° E to 146° E)", min_value=122.0, max_value=145.8, value=140.07, step=0.000001, format="%0.6f")
    # inp_depth = st.number_input("Depth (2.0 km to 442.0 km)", min_value=2.0, max_value=442.0, value=10.00)
    inp_month = st.selectbox("Select a month:", months)
    submit = st.button("Predict")

with col2:
    st.subheader("Prediction")

    if submit:
        input_data = np.array([[inp_latitude, inp_longitude, inp_month]])
        prediction = np.round(model.predict(input_data), 2)

        # Extracting location details, like city and country
        # Initialize the geolocator
        geolocator = Nominatim(user_agent="city_country_finder")
        # Reverse geocoding
        location = geolocator.reverse((inp_latitude, inp_longitude), exactly_one=True,
                                      language='en')  # Replace with your lat, lon
        # Extract city and country
        if location and 'address' in location.raw:
            address = location.raw['address']
            city = address.get('city') or address.get('town') or address.get('village') or address.get(
                'municipality') or 'Unknown'
            country = address.get('country', 'Unknown')
        else:
            city, country = 'Unknown', 'Unknown'

        # Get month name from month number given by user
        if inp_month == 1:
            month_name = "January"
        elif inp_month == 2:
            month_name = "February"
        elif inp_month == 3:
            month_name = "March"
        elif inp_month == 4:
            month_name = "April"
        elif inp_month == 5:
            month_name = "May"
        elif inp_month == 6:
            month_name = "June"
        elif inp_month == 7:
            month_name = "July"
        elif inp_month == 8:
            month_name = "August"
        elif inp_month == 9:
            month_name = "September"
        elif inp_month == 10:
            month_name = "October"
        elif inp_month == 11:
            month_name = "November"
        elif inp_month == 12:
            month_name = "December"

        st.success(f"Earthquake of magnitude **{prediction}** is predicted at **{city}**, **{country}** in "
                   f"**{month_name}**")

        if prediction <= 4.0:
            st.info(f"**Low Magnitude (Minor)** \n- Usually not felt. \n- Little to no damage.")
        elif 4.0 < prediction <= 5.9:
            st.warning(f"**Moderate Magnitude** \n- Can cause minor to moderate damage, especially near the "
                       f"epicenter. \n- Buildings may shake, and some structures could develop cracks")
        else:
            st.error(f"**High Magnitude (Major)** \n- Can cause significant to catastrophic damage, particularly in "
                     f"populated or poorly constructed areas. \n- Aftershocks and tsunamis may also occur.")

with col3:
    if submit:
        location = pd.DataFrame({'lat': [inp_latitude], 'lon': [inp_longitude]})
        st.map(location, zoom=6)

