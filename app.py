# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # 파일 존재 여부 확인용

# 경고 메시지 무시
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="수면 건강 예측", layout="centered")

st.title("수면의 질 예측")
st.markdown("사용자의 정보를 입력하여 수면의 질을 예측합니다.")

# --- 모델 및 LabelEncoder 로드 ---
@st.cache_resource # 앱 시작 시 한 번만 로드하도록 캐싱
def load_resources():
    try:
        model = joblib.load('rf_model.pkl')
        target_le = joblib.load('target_label_encoder.pkl')

        # 각 범주형 컬럼별 LabelEncoder 로드
        # 학습 시 사용된 컬럼명과 일치해야 합니다.
        # 데이터셋을 확인하고 필요한 컬럼을 여기에 추가하세요.
        encoders = {}
        categorical_cols_to_load = [
            'gender', # Gender
            'occupation', # Occupation
            'bmi_category', # BMI Category
            'sleep_disorder' # Sleep Disorder
        ]
        for col_name in categorical_cols_to_load:
            # 파일명을 소문자_로 변환한 이름과 일치하도록 합니다.
            file_path = f'{col_name}_label_encoder.pkl'
            if os.path.exists(file_path):
                encoders[col_name] = joblib.load(file_path)
            else:
                st.error(f"Error: {file_path} 파일을 찾을 수 없습니다. 'train_model.py'를 실행하여 인코더를 생성했는지 확인해주세요.")
                st.stop()
        return model, target_le, encoders
    except FileNotFoundError as e:
        st.error(f"필수 파일을 찾을 수 없습니다: {e}. 'train_model.py'를 실행하여 모델과 인코더를 생성했는지 확인해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"리소스 로딩 중 오류 발생: {e}")
        st.stop()

rf_model, y_le, encoders = load_resources()

# --- 사용자 입력 UI ---
st.header("사용자 정보 입력")

# 레이아웃을 위해 컬럼 사용
col1, col2 = st.columns(2)

with col1:
    age = st.slider("나이", 18, 100, 30, key='age')
    gender = st.selectbox("성별", ['Male', 'Female'], key='gender')
    sleep_duration = st.slider("수면 시간 (시간)", 4.0, 10.0, 7.0, 0.1, key='sleep_duration')
    physical_activity_level = st.slider("신체 활동 수준 (분)", 0, 300, 60, key='physical_activity_level')
    stress_level = st.slider("스트레스 수준 (1-10)", 1, 10, 5, key='stress_level')

with col2:
    # 데이터셋의 실제 직업군을 파악하여 리스트에 넣어주세요.
    # train_model.py에서 'occupation_label_encoder.pkl'을 로드한 후
    # `encoders['occupation'].classes_.tolist()`를 통해 실제 사용된 직업군 리스트를 얻을 수 있습니다.
    occupation_options = encoders['occupation'].classes_.tolist() if 'occupation' in encoders else ['Software Engineer', 'Doctor', 'Nurse', 'Teacher', 'Sales Representative', 'Scientist', 'Lawyer', 'Accountant', 'Manager', 'Engineer'] # 예시
    occupation = st.selectbox("직업", occupation_options, key='occupation')

    # 데이터셋의 실제 BMI 카테고리를 파악하여 리스트에 넣어주세요.
    bmi_category = st.selectbox("BMI 카테고리", ['Normal', 'Overweight', 'Obese', 'Normal Weight'], key='bmi_category')
    heart_rate = st.slider("심박수 (bpm)", 50, 100, 70, key='heart_rate')
    daily_steps = st.slider("일일 걸음 수", 1000, 10000, 5000, key='daily_steps')
    sleep_disorder = st.selectbox("수면 장애", ['None', 'Sleep Apnea', 'Insomnia'], key='sleep_disorder')

# 'Blood Pressure' 컬럼이 모델 학습에 사용되었다면, 여기에 입력 필드를 추가해야 합니다.
# train_model.py에서 'Mean BP'로 변환했다면, 'Mean BP'를 예측 입력으로 받거나,
# Systolic BP, Diastolic BP를 받아서 Streamlit 앱 내에서 'Mean BP'를 계산해야 합니다.
# 예시:
# mean_bp = st.slider("평균 혈압", 60, 150, 90, key='mean_bp') # Blood Pressure 대신 Mean BP 사용 가정

if st.button("수면의 질 예측하기", help="입력된 정보로 수면의 질을 예측합니다."):
    with st.spinner("예측 중입니다..."):
        try:
            # 입력 데이터 전처리 (학습 시와 동일한 방식)
            # 각 컬럼별 LabelEncoder를 사용하여 변환
            gender_encoded = encoders['gender'].transform([gender])[0]
            occupation_encoded = encoders['occupation'].transform([occupation])[0]
            bmi_category_encoded = encoders['bmi_category'].transform([bmi_category])[0]
            sleep_disorder_encoded = encoders['sleep_disorder'].transform([sleep_disorder])[0]

            # 예측을 위한 데이터프레임 생성
            # 컬럼 순서와 이름은 모델 학습 시 사용된 X의 컬럼 순서 및 이름과 정확히 일치해야 합니다.
            # train_model.py에서 X를 생성한 후 df.columns를 확인하여 정확한 순서를 알아내세요.
            # 예시 (train_model.py의 X.columns 순서에 따라 조정 필요):
            # df.drop(['Person ID', 'Quality of Sleep', 'Blood Pressure'], axis=1)
            # Blood Pressure가 'Mean BP'로 대체되었다면 다음과 같이 컬럼을 구성해야 합니다.
            input_data = pd.DataFrame([[
                age,
                gender_encoded,
                sleep_duration,
                occupation_encoded, # 'Occupation' 컬럼의 위치 확인
                stress_level,
                bmi_category_encoded,
                heart_rate,
                daily_steps,
                sleep_disorder_encoded,
                physical_activity_level, # 'Physical Activity Level' 컬럼의 위치 확인
                # Mean BP가 있다면 여기에 추가: mean_bp
            ]], columns=[
                'Age', '
                'Gender',
                'Sleep Duration',
                'Occupation',
                'Stress Level',
                'BMI Category',
                'Heart Rate',
                'Daily Steps',
                'Sleep Disorder',
                'Physical Activity Level',
                # 'Mean BP' # 만약 모델에 'Mean BP'가 포함되어 있다면
            ])

            # 컬럼 순서 일치 확인 (매우 중요)
            # 학습 데이터의 컬럼 순서를 정확히 따르도록 합니다.
            # rf_model.feature_names_in_ 는 학습 시 사용된 컬럼 순서를 알려줍니다.
            if hasattr(rf_model, 'feature_names_in_'):
                input_data = input_data[rf_model.feature_names_in_]
            else:
                st.warning("경고: 모델의 feature_names_in_ 속성을 찾을 수 없습니다. 입력 컬럼 순서를 수동으로 확인해야 합니다.")

            # 예측 수행
            prediction_encoded = rf_model.predict(input_data)
            predicted_quality_of_sleep = y_le.inverse_transform(prediction_encoded)[0]

            st.success(f"예측된 수면의 질은 **{predicted_quality_of_sleep}** 입니다!")
            st.balloons() # 예측 성공 시 풍선 효과

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.info("입력 데이터의 형식과 모델 학습 시 사용된 컬럼 구성이 일치하는지 확인해주세요.")
            st.write("--- 디버깅 정보 ---")
            st.write(f"입력 데이터 (전처리 후):")
            st.dataframe(input_data)
            #st.write(f"모델의 예상 입력 컬럼 순서: {rf_model.feature_names_in_}")