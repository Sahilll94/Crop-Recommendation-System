import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")  # Ensure this file is in your working directory

# Features (N, P, K, temperature, humidity, pH, rainfall) and target (crop label)
X = df.drop("label", axis=1)
y = df["label"]

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "crop_recommendation_model.pkl")

print("✅ Model trained and saved successfully!")
