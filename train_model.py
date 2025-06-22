# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

print("1. 데이터 불러오기...")
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

print("2. 데이터 전처리...")
# Person ID 컬럼 제거 (예측에 불필요)
if 'Person ID' in df.columns:
    df = df.drop(columns=['Person ID'])

# 결측치 및 이상치 처리 (현재 데이터셋은 결측치가 없는 것으로 알려져 있습니다. Inf 값도 없다고 가정)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Blood Pressure 컬럼 분리 및 평균 계산 (문자열 형태 "XXX/YYY"를 숫자로 변환)
# 만약 'Blood Pressure' 컬럼이 없다면 이 부분을 주석 처리하거나 제거해야 합니다.
if 'Blood Pressure' in df.columns:
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df['Mean BP'] = (df['Systolic BP'] + df['Diastolic BP']) / 2
    df = df.drop(columns=['Blood Pressure', 'Systolic BP', 'Diastolic BP'])

print("3. 타겟 변수 정의...")
target = 'Quality of Sleep'
X = df.drop(columns=[target])
y = df[target]

print("4. 범주형 변수 인코딩 및 저장...")
# 각 범주형 변수에 대해 LabelEncoder를 적용하고 저장
categorical_cols = X.select_dtypes(include='object').columns

# 'Occupation' 컬럼의 고유 값들이 너무 많아 모든 값을 포함하는 인코더를 학습 시 생성해야 합니다.
# 실제 데이터셋의 모든 직업군을 포함하도록 학습 데이터를 사용합니다.
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # 주의: fit_transform 대신 fit만 사용하여 모든 고유 값을 학습시키고,
    # 변환은 나중에 수행합니다. 이는 전체 데이터셋의 가능한 모든 카테고리를 포함하기 위함입니다.
    le.fit(X[col])
    X[col] = le.transform(X[col])
    encoders[col] = le
    joblib.dump(le, f'{col.lower().replace(" ", "_")}_label_encoder.pkl')
    print(f" - {col} LabelEncoder 저장 완료.")

# y 타겟 변수 인코딩 및 저장
y_le = LabelEncoder()
y = y_le.fit_transform(y)
joblib.dump(y_le, 'target_label_encoder.pkl')
print(f" - {target} LabelEncoder 저장 완료.")

print("5. SMOTE 적용 (k_neighbors=2)...")
# k_neighbors는 클래스의 최소 샘플 수보다 작아야 합니다.
# 안전을 위해 k_neighbors를 조절할 수 있습니다.
min_samples_per_class = y_resampled_check = pd.Series(y).value_counts().min()
smote_k_neighbors = min(2, min_samples_per_class - 1) if min_samples_per_class > 1 else 1
if smote_k_neighbors < 1: smote_k_neighbors = 1 # 최소 1

smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f" - SMOTE 적용 완료. (k_neighbors={smote_k_neighbors})")

print("6. 학습/테스트 분할...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
# stratify=y_resampled 추가하여 클래스 비율 유지

print("7. 모델 학습 (RandomForestClassifier)...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10) # 예시 하이퍼파라미터
rf_model.fit(X_train, y_train)
print(" - 모델 학습 완료.")

print("8. 예측 및 평가...")
y_pred = rf_model.predict(X_test)

print("\n--- 모델 평가 결과 ---")
print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:\n", classification_report(y_test, y_pred, target_names=y_le.classes_.astype(str)))

print("9. 교차검증...")
cv_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5, scoring='accuracy') # cv=5로 변경
print("교차검증 평균 정확도:", cv_scores.mean())
print("교차검증 스코어:", cv_scores)

print("10. 모델 저장...")
joblib.dump(rf_model, 'rf_model.pkl')
print(" - rf_model.pkl 저장 완료.")

print("\n모델 학습 및 저장이 완료되었습니다!")
print("웹 앱 배포를 위해 'rf_model.pkl'과 모든 '_label_encoder.pkl' 파일들이 준비되었습니다.")