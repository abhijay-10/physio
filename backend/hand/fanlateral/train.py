import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATASET
# ==========================================
dataset_file = "fan_lateral_dataset.csv"

try:
    df = pd.read_csv(dataset_file)
    print(f"✅ Dataset loaded. Found {len(df)} rows.")
except FileNotFoundError:
    print("❌ CSV file not found. Please collect data first.")
    exit()

# ==========================================
# 2. PREPROCESS DATA
# ==========================================
# Features (X) are all columns except 'label'
X = df.drop("label", axis=1)

# Target (y) is the 'label' column
y = df["label"]

# Encode text labels into numbers (0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. TRAIN RANDOM FOREST MODEL
# ==========================================
print("Training model... please wait.")

# We use 100 trees (n_estimators) for a good balance of speed and accuracy
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATE MODEL
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("===================================\n")
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE MODELS FOR STREAMLIT
# ==========================================
model_filename = "fanhand_model.pkl"
encoder_filename = "fanlabel_encoder.pkl"

joblib.dump(model, model_filename)
joblib.dump(label_encoder, encoder_filename)

print(f"✅ Model saved as: {model_filename}")
print(f"✅ Encoder saved as: {encoder_filename}")
print("\nYou can now run your Streamlit app!")