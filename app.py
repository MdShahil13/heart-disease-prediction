import streamlit as st
import pandas as pd
import joblib


# Load model files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


st.title("Heart Disease Prediction App ❤️‍🩹")
st.markdown("Provide patient details to predict heart disease risk.")


age = st.number_input("Age", min_value=18, max_value=100, value=30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    ["Yes", "No"]
)

rest_ecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "ST Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)



if st.button("Predict"):


    # Create input with same features as training

    raw_input = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS": 1 if fbs == "Yes" else 0,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,


        # Sex
        "Sex_M": 1 if sex == "Male" else 0,


        # Exercise Angina
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,


        # Chest Pain default values
        "ChestPainType_ATA": 0,
        "ChestPainType_NAP": 0,
        "ChestPainType_TA": 0,


        # Rest ECG default values
        "RestingECG_Normal": 0,
        "RestingECG_ST": 0,


        # Slope default values
        "ST_Slope_Flat": 0,
        "ST_Slope_Up": 0
    }



    # Chest Pain Encoding

    if cp == "Atypical Angina":
        raw_input["ChestPainType_ATA"] = 1

    elif cp == "Non-Anginal Pain":
        raw_input["ChestPainType_NAP"] = 1

    elif cp == "Typical Angina":
        raw_input["ChestPainType_TA"] = 1



    # Rest ECG Encoding

    if rest_ecg == "Normal":
        raw_input["RestingECG_Normal"] = 1

    elif rest_ecg == "ST-T Wave Abnormality":
        raw_input["RestingECG_ST"] = 1



    # ST Slope Encoding

    if slope == "Flat":
        raw_input["ST_Slope_Flat"] = 1

    elif slope == "Upsloping":
        raw_input["ST_Slope_Up"] = 1



    input_df = pd.DataFrame([raw_input])

    input_df = input_df.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )



    # Scaling

    input_scaled = scaler.transform(input_df)


    # Prediction

    prediction = model.predict(input_scaled)[0]


    if prediction == 1:
        st.error("💔 High Risk of Heart Disease")

    else:
        st.success("💚 Low Risk of Heart Disease")