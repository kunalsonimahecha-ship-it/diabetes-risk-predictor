import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and train model
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# App UI
st.title("Diabetes Risk Predictor")
st.write("Enter patient details below:")

pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose Level", 44, 199, 120)
bp = st.slider("Blood Pressure", 24, 122, 72)
skinthickness = st.slider("Skin Thickness", 7, 99, 29)
insulin = st.slider("Insulin", 14, 846, 141)
bmi = st.slider("BMI", 18.2, 67.1, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47)
age = st.slider("Age", 21, 81, 33)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skinthickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[:,1]
    
    if prob >= 0.4:
        st.error(f"High Diabetes Risk — Probability: {prob[0]:.2%}")
    else:
        st.success(f"Low Diabetes Risk — Probability: {prob[0]:.2%}")
