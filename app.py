import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 & 인코더 불러오기
rf_model = joblib.load('rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("수면의 질 예측 웹앱")

# 사용자 입력 받기
gender = st.selectbox("성별", ['Male', 'Female'])
age = st.slider("나이", 18, 100, 30)
occupation = st.selectbox("직업", ['Doctor', 'Engineer', 'Teacher', 'Nurse', 'Lawyer', 'Scientist', 'Salesperson', 'Accountant', 'Software Engineer', 'Other'])
sleep_duration = st.slider("수면 시간 (시간)", 0.0, 12.0, 7.0, step=0.5)
physical_activity = st.slider("신체 활동 수준 (1~10)", 1, 10, 5)
stress_level = st.slider("스트레스 수준 (1~10)", 1, 10, 5)
bmi_category = st.selectbox("BMI 카테고리", ['Normal', 'Overweight', 'Obese', 'Underweight'])
blood_pressure = st.selectbox("혈압", ['Normal', 'High', 'Low'])
heart_rate = st.slider("심박수", 40, 130, 70)
daily_steps = st.number_input("일일 걸음 수", min_value=0, max_value=30000, value=5000)
sleep_disorder = st.selectbox("수면 장애", ['None', 'Insomnia', 'Sleep Apnea'])

# 예측
if st.button("예측하기"):
    # 입력 데이터 구성
    input_dict = {
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'BMI Category_Normal': 1 if bmi_category == 'Normal' else 0,
        'BMI Category_Overweight': 1 if bmi_category == 'Overweight' else 0,
        'BMI Category_Obese': 1 if bmi_category == 'Obese' else 0,
        'BMI Category_Underweight': 1 if bmi_category == 'Underweight' else 0,
        'Sleep Disorder_None': 1 if sleep_disorder == 'None' else 0,
        'Sleep Disorder_Insomnia': 1 if sleep_disorder == 'Insomnia' else 0,
        'Sleep Disorder_Sleep Apnea': 1 if sleep_disorder == 'Sleep Apnea' else 0,
    }

    # 직업 one-hot 인코딩
    for job in ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Nurse', 'Other', 'Salesperson', 'Scientist', 'Software Engineer', 'Teacher']:
        input_dict[f'Occupation_{job}'] = 1 if occupation == job else 0

    input_df = pd.DataFrame([input_dict])
    prediction = rf_model.predict(input_df)
    predicted_quality = label_encoder.inverse_transform(prediction)[0]

    st.success(f"예측된 수면의 질: {predicted_quality}")

