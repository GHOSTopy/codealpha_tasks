import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Page Title
st.set_page_config(page_title="🩺 Disease Prediction", layout="centered")
st.title("🩺 Disease Prediction from Medical Data")
st.write("This app predicts the likelihood of **Heart Disease** based on patient data.")

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

# Display data
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# Feature selection
X = df.drop('target', axis=1)
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar Input
st.sidebar.header("Enter Patient Data")

def user_input():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex (0 = female, 1 = male)', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (0–3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting BP', 94, 200, 130)
    chol = st.sidebar.slider('Cholesterol', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 (1 = true; 0 = false)', [0, 1])
    restecg = st.sidebar.slider('Rest ECG (0–2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise-induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of ST (0–2)', 0, 2, 1)
    ca = st.sidebar.slider('No. of Major Vessels (0–4)', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)', [1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])

user_df = user_input()

# Prediction
prediction = model.predict(user_df)
prediction_proba = model.predict_proba(user_df)

# Result
st.subheader("Prediction Result")
st.write("🧠 The patient is **{}** Heart Disease.".format(
    "at risk of" if prediction[0] == 1 else "not at risk of"
))
st.write("Prediction Probability:", prediction_proba)

# Accuracy
st.subheader("Model Evaluation")
st.write(f"✅ Accuracy: {accuracy:.2f}")

# Classification Report
st.text("📊 Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

