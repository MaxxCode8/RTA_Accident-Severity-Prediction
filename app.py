import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_enc_day, categorical_enc

# Load Trained Model ...
model = joblib.load(r'./Model/random_forest_14.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚗🚧🚶‍♀️", layout="centered")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_juction = ['Crossing', 'No junction', 'O Shape','Other','T Shape','Unknown','X Shape','Y Shape']
options_service_yr = ["1-2yr", "2-5yrs", "5-10yrs", "Above 10yrs", "Below 1yr", "Unknown"]
options_casualty_age = ["18-30", "31-50", "Under 5", "Over 51", "Unknown", "Under 18", "NA"]
options_collision_type = ["Collision with animals", "Collision with pedestrians", "Collision with roadside objects", "Collision with roadside-parked vehicles",
                          "Fall from vehicles", 'Other', "Rollover", "Unknown", "Vehicle with vehicle collision"]



st.markdown("<h2 style='text-align: center;'> 🚶🚓 Accident Severity Prediction App 🚗🚧🏥</h2>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Report the Observations of the Accident Consciously:")
        
        hour = st.slider("Accident Pick-up Hour: ", 0, 23, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        casualties = st.slider("Select No of Casulaties: ", 1, 8, value=0, format="%d")
        accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Select Vehicles Involved: ", 1, 7, value=0, format="%d")
        vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        junction = st.selectbox("Select Junction: ", options=options_juction)
        service_yr = st.selectbox("Service Year: ", options=options_service_yr)
        casualty_age = st.selectbox("Casualty Age: ", options=options_casualty_age)
        collision_type = st.selectbox("Select Collision Type: ", options=options_collision_type)
        # light_condition = st.selectbox("Select Light Condition: ", options=options_light_condition)
        
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = ordinal_enc_day(day_of_week)
        driver_age = categorical_enc(driver_age, options_age)
        driving_experience = categorical_enc(driving_experience, options_driver_exp) 
        accident_cause = categorical_enc(accident_cause, options_cause)
        vehicle_type = categorical_enc(vehicle_type, options_vehicle_type)
        accident_area =  categorical_enc(accident_area, options_acc_area)
        lanes = categorical_enc(lanes, options_lanes)
        junction = categorical_enc(junction, options_juction)
        service_yr = categorical_enc(service_yr, options_service_yr)
        casualty_age = categorical_enc(casualty_age, options_casualty_age)
        collision_type = categorical_enc(collision_type, options_collision_type)
       
        
        data = driver_age + driving_experience + vehicle_type + service_yr + accident_area + lanes + junction + collision_type + \
            casualty_age + accident_cause + [hour] + [casualties] + [day_of_week] + [vehicles_involved]
        data = np.array(data).reshape(1, -1)
        pred = get_prediction(data=data, model=model)
        
        st.markdown(f"<h2 style='text-align: center;'> Accident Severity: {pred} 🏥</h2>", unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()