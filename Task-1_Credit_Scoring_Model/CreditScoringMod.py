import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credit Scoring Model", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #b2d8d8;
    }
    .stApp {
        background-color: #b2d8d8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💳 Credit Scoring Prediction")
st.subheader("Enter details to check creditworthiness.")

age = st.number_input("Age", min_value=18, max_value=80, value=25)
income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
loan = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=20000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0, max_value=100, value=30)
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])

employment_map = {"Employed": 1, "Unemployed": 0, "Self-Employed": 2}
employment_encoded = employment_map[employment]

np.random.seed(42)
data_size = 1000
X = pd.DataFrame({
    "Age": np.random.randint(18, 80, data_size),
    "Annual Income": np.random.randint(10000, 200000, data_size),
    "Loan Amount": np.random.randint(1000, 100000, data_size),
    "Credit Score": np.random.randint(300, 850, data_size),
    "Debt-to-Income Ratio": np.random.randint(10, 60, data_size),
    "Employment Status": np.random.choice([0, 1, 2], data_size)
})

y = np.concatenate((
    np.ones(data_size // 2),
    np.zeros(data_size // 2)
))
np.random.shuffle(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

if st.button("Predict"):
    user_data = np.array([[age, income, loan, credit_score, dti, employment_encoded]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Not Creditworthy")
    else:
        st.success("✅ Low Risk: Creditworthy")

# Run this script using the command: streamlit run script_name.py
