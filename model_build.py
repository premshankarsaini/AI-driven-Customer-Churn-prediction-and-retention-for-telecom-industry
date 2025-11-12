import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# 🧩 Step 1: Dummy dataset create kar rahe hain
data = {
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'tenure': [5, 20, 10, 30, 15, 8, 25, 40, 2, 12],
    'OnlineSecurity': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'OnlineBackup': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'DeviceProtection': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'TechSupport': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Contract': ['Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year'],
    'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'MonthlyCharges': [50, 80, 60, 70, 90, 65, 55, 85, 45, 75],
    'TotalCharges': [300, 1600, 600, 2100, 900, 520, 1200, 3400, 150, 850],
    'Churn': [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]   # Target column
}

df = pd.DataFrame(data)

# 🧩 Step 2: Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 🧩 Step 3: Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🧩 Step 4: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🧩 Step 5: Save the model
pickle.dump(model, open('Model.sav', 'wb'))

print("✅ Model.sav file successfully created!")