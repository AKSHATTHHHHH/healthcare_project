import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("healthcare_project/merged_clean_health_dataset.csv") 
print("Columns in CSV:\n", df.columns)

# Optional: Preview first few rows
print("\nSample data:\n", df.head())

# Clean missing/infinite values
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Replace infinities with NaN (if any)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ Add this block now (Line ~23)
from sklearn.preprocessing import LabelEncoder
label_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Drop non-numeric or irrelevant columns (optional, based on your CSV)
columns_to_drop = ['Name', 'Doctor', 'Hospital', 'Insurance Provider', 
                   'Date of Admission', 'Discharge Date']
for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values (if any remain)
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Add Rule-Based AI Diagnosis
def ai_diagnosis(row):
    if row['chol'] > 240:
        return "High Cholesterol"
    elif row['thalach'] < 100:
        return "Low Heart Rate - Possible Risk"
    elif row['fbs'] == 1:
        return "Possible Diabetes"
    else:
        return "Normal"

# Apply diagnosis rules to the full dataset
df['Diagnosis'] = df.apply(ai_diagnosis, axis=1)

# Preview AI diagnosis
print("\nSample AI Diagnoses:")
print(df[['age', 'chol', 'thalach', 'fbs', 'Diagnosis']].head())


import matplotlib.pyplot as plt
import seaborn as sns

# Set chart style
sns.set(style="whitegrid")

# 📊 1. Bar Chart: Count of AI Diagnoses
plt.figure(figsize=(8, 5))
sns.countplot(x='Diagnosis', data=df, palette='Set2')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis Category')
plt.ylabel('Number of Patients')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 📊 2. Pie Chart: Diagnosis Share
plt.figure(figsize=(6, 6))
df['Diagnosis'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title('Diagnosis Proportion')
plt.ylabel('')
plt.tight_layout()
plt.show()