import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load cleaned dataset
data = pd.read_csv("dataset/diabetes_cleaned.csv")

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Healthcare Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)
# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🏥 Healthcare AI")

st.sidebar.markdown("""
### Navigation

Select any page to explore the project.
""")

st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Information",
        "📈 Visualizations",
        "📈 Model Performance",
        "🤖 Disease Prediction",
        "ℹ️ About Project"
    ]
)

if page == "🏠 Home":

    st.title("🏥 Healthcare Data Analysis and Disease Prediction System")

    st.markdown("""
Welcome to the **Healthcare Data Analysis and Disease Prediction System**.

This project uses **Machine Learning** to predict whether a patient is likely to have **Diabetes** based on important medical parameters.

The application demonstrates the complete Data Science workflow:
- 📊 Data Analysis
- 🧹 Data Preprocessing
- 📈 Data Visualization
- 🤖 Machine Learning
- 🌐 Interactive Web Application
""")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", "768")

    with col2:
        st.metric("Input Features", "8")

    with col3:
        st.metric("Best Accuracy", "75%")

    st.divider()

    st.success("✅ Best Model: Random Forest Classifier")

    st.info("""
Random Forest was selected as the final model because it achieved the best overall performance while maintaining a good balance between accuracy and prediction capability.
""")
# ------------------------------
# Dataset Information Page
# ------------------------------
elif page == "📊 Dataset Information":

    st.title("📊 Dataset Information")

    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", data.shape[0])

    with col2:
        st.metric("Columns", data.shape[1])

    with col3:
        st.metric("Features", data.shape[1]-1)
    
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Diabetic Patients", int(data["Outcome"].sum()))

    with col2:
        st.metric("Non-Diabetic Patients", int((data["Outcome"] == 0).sum()))

    st.divider()

    st.subheader("Dataset Columns")

    st.write(data.columns.tolist())

    st.divider()

    st.subheader("First 5 Records")

    st.dataframe(data.head(), use_container_width="stretch")

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(data.describe(), use_container_width="stretch")

    # ------------------------------
# Visualizations Page
# ------------------------------

elif page == "📈 Visualizations":

    st.title("📈 Data Visualizations")

    st.markdown("Explore the visual analysis of the Diabetes Dataset.")

    st.divider()

    st.subheader("Diabetes Distribution")
    st.image("screenshots/diabetes_distribution.png", use_container_width="stretch")

    st.divider()

    st.subheader("Glucose Distribution")
    st.image("screenshots/glucose_histogram.png", use_container_width="stretch")

    st.divider()

    st.subheader("BMI Distribution")
    st.image("screenshots/bmi_histogram.png", use_container_width="stretch")

    st.divider()

    st.subheader("Age Distribution")
    st.image("screenshots/age_histogram.png", use_container_width="stretch")

    st.divider()

    st.subheader("Insulin Distribution")
    st.image("screenshots/insulin_histogram.png", use_container_width="stretch")

    st.divider()

    st.subheader("Correlation Heatmap")
    st.image("screenshots/correlation_heatmap.png", use_container_width="stretch")

    st.divider()

    st.subheader("Feature Importance")

    st.image(
        "screenshots/feature_importance.png",
        use_container_width="stretch"
    )

 # ------------------------------
# Model Performance Page
# ------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write("Comparison of all machine learning models used for diabetes prediction.")

    st.divider()

    # Performance Metrics
    performance = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [0.75, 0.72, 0.75],
        "Precision": [0.67, 0.60, 0.64],
        "Recall": [0.62, 0.64, 0.67],
        "F1-Score": [0.64, 0.62, 0.65]
    })

    st.subheader("📋 Model Comparison Table")
    st.dataframe(performance, use_container_width="stretch")

    st.divider()

    st.subheader("📊 Accuracy Comparison")
    st.bar_chart(
        performance.set_index("Model")["Accuracy"]
    )

    st.divider()

    st.subheader("📈 Precision, Recall and F1-Score")

    metrics = performance.set_index("Model")[["Precision", "Recall", "F1-Score"]]

    st.bar_chart(metrics)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Accuracy",
            "75%"
        )

    with col2:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    with col3:
        st.metric(
            "Dataset Size",
            "768"
        )

    st.success("""
🏆 **Final Selected Model: Random Forest Classifier**

Reason:
- High Accuracy
- Good Recall
- Better Generalization
- Robust against overfitting
""")   


# ------------------------------
# Disease Prediction Page
# ------------------------------

elif page == "🤖 Disease Prediction":

    st.title("🤖 Diabetes Prediction System")

    st.write("Enter the patient's health details below.")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Predict Disease"):

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        confidence = max(probability) * 100

        st.divider()

        if prediction == 1:

            st.error("🔴 Prediction: Diabetic")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.warning("""
### 🩺 Health Recommendations

• Consult a healthcare professional.

• Monitor blood glucose regularly.

• Reduce sugar intake.

• Exercise at least 30 minutes daily.

• Maintain a healthy weight.
""")

        else:

            st.success("🟢 Prediction: Non-Diabetic")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.info("""
### 🌿 Healthy Lifestyle Tips

• Continue regular exercise.

• Maintain a balanced diet.

• Drink enough water.

• Schedule regular health checkups.

• Maintain a healthy BMI.
""")

        # -------------------------
        # Generate Prediction Report
        # -------------------------

        current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        report = f"""
Healthcare Data Analysis and Disease Prediction Report

----------------------------------------

Date & Time: {current_date}

Prediction:
{"Diabetic" if prediction == 1 else "Non-Diabetic"}

Confidence:
{confidence:.2f}%

----------------------------------------

Patient Details

Pregnancies: {pregnancies}

Glucose: {glucose}

Blood Pressure: {blood_pressure}

Skin Thickness: {skin_thickness}

Insulin: {insulin}

BMI: {bmi}

Diabetes Pedigree Function: {diabetes_pedigree}

Age: {age}

----------------------------------------

Generated using Streamlit and Machine Learning.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="Diabetes_Prediction_Report.txt",
            mime="text/plain"
        )


# ------------------------------
# About Project Page
# ------------------------------

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.markdown("""
## 🏥 Healthcare Data Analysis and Disease Prediction System

### Project Objective

The objective of this project is to analyze healthcare data and predict whether a patient is likely to have diabetes using Machine Learning algorithms.

The system helps users understand diabetes prediction based on important health parameters and demonstrates the complete data science workflow.

---

### Dataset

Dataset Name:
Pima Indians Diabetes Dataset

Total Records:
768

Total Features:
8 Input Features + 1 Target Variable

Target Variable:
Outcome
- 0 = Non-Diabetic
- 1 = Diabetic

---

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

---

### Machine Learning Algorithms

✔ Logistic Regression

✔ Decision Tree

✔ Random Forest

---

### Best Model

Random Forest Classifier

Accuracy:
75%

---

### Developed By

Jebachristina M

Bachelor of Technology

Artificial Intelligence and Data Science

    """)
    st.warning("""
### Disclaimer

This project is developed for educational purposes only.

It should not be used as a substitute for professional medical advice or diagnosis.
""")

st.markdown("---")
st.caption("Healthcare Data Analysis and Disease Prediction System | Developed using Streamlit and Machine Learning")

   
