import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

CSV_PATH = "pa_hand_data.csv"

# Load WITHOUT header
df = pd.read_csv(CSV_PATH, header=None)

print("📊 Raw shape:", df.shape)

# Assign correct column names (8 columns!)
df.columns = ["w_z", "i_z", "m_z", "r_z", "p_z", "spread", "extra", "label"]

print("✅ Columns fixed:", df.columns)

# Clean labels
df["label"] = df["label"].astype(str).str.lower().str.strip()

# Convert labels
df["label"] = df["label"].map({
    "right": 1,
    "left": 1,
    "wrong": 0
})

# Drop bad rows
df = df.dropna()

# Features & labels
X = df.drop("label", axis=1)
y = df["label"]

print("📊 Label distribution:")
print(y.value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"🎯 Accuracy: {accuracy:.2f}")

# Save
joblib.dump(model, "pa_model.pkl")

print("✅ Model saved successfully!")