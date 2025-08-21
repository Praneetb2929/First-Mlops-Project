import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Safe print with fallback (no emoji if Windows can't handle it)
def safe_print(message):
    try:
        print("✅", message)
    except UnicodeEncodeError:
        print("[OK]", message)

safe_print(f"Columns: {df.columns.tolist()}")

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "diabetes_model.pkl")
safe_print("Model saved as diabetes_model.pkl")

