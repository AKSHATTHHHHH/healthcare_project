import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI setup
st.set_page_config(page_title="Heart Health AI", layout="centered")
st.title("🧠 AI-Powered Heart Diagnosis System")
st.write("This system uses ML + expert rules to assess patient heart health.")

# ✅ Data loading and training function
@st.cache_data
def load_and_train():
    df = pd.read_csv("healthcare_project/merged_clean_health_dataset.csv")

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    # Drop unnecessary text columns
    drop_cols = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 'Date of Admission', 'Discharge Date']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Convert all object (text) columns to numeric (e.g., 'Male' -> 1, 'Female' -> 0)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    if 'target' not in df.columns:
        st.error("❌ 'target' column not found!")
        return None, None, None, None, None, None

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, X, y, df, acc

# 🔄 Load model and data
model, scaler, X_full, y_full, df_full, accuracy = load_and_train()

if model:
    st.success(f"✅ Model trained with accuracy: {accuracy:.2%}")

    # --- Patient Input Section ---
    st.subheader("📋 Enter Patient Data for Diagnosis")
    input_data = {}
    columns_for_input = df_full.drop(columns='target').columns.tolist()

    for i, col in enumerate(columns_for_input):
        input_data[col] = st.number_input(f"{col}", value=0.0, key=f"input_{i}")

    if st.button("🔍 Diagnose"):
        user_df = pd.DataFrame([input_data])
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]

        # Rule-based diagnosis
        chol = input_data.get('chol', 0)
        thalach = input_data.get('thalach', 0)
        fbs = input_data.get('fbs', 0)

        if chol > 240:
            rule_diag = "High Cholesterol"
        elif thalach < 100:
            rule_diag = "Low Heart Rate - Risk"
        elif fbs == 1:
            rule_diag = "Possible Diabetes"
        else:
            rule_diag = "Normal"

        # Display results
        st.subheader("🩺 AI Diagnosis Result:")
        if prediction == 1:
            st.error("⚠️ ML Diagnosis: Patient may have heart disease.")
        else:
            st.success("✅ ML Diagnosis: Heart is likely healthy.")

        st.info(f"💡 Rule-Based Insight: {rule_diag}")

    # --- Visual Summary Charts ---
    st.subheader("📊 Diagnosis Distribution Overview")

    # Add 'Diagnosis' column to dataframe if missing
    if 'Diagnosis' not in df_full.columns:
        df_full['Diagnosis'] = df_full.apply(
            lambda row: "High Cholesterol" if row['chol'] > 240 else
                        "Low Heart Rate - Risk" if row['thalach'] < 100 else
                        "Possible Diabetes" if row['fbs'] == 1 else
                        "Normal", axis=1
        )

    sns.set(style="whitegrid")

    # Bar chart
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df_full, x='Diagnosis', palette='Set2', ax=ax1)
    ax1.set_title('Diagnosis Distribution')
    st.pyplot(fig1)

    # Pie chart
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    df_full['Diagnosis'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'), ax=ax2
    )
    ax2.set_ylabel('')
    ax2.set_title('Diagnosis Proportion')
    st.pyplot(fig2)