import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. LOAD DATASET
# ==========================================
DATASET_PATH = "lordotic_dataset.csv"  # Ensure this matches your file name

print("--- 📂 Loading Dataset ---")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found. Make sure the file is in this folder.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Features (X): All coordinate columns
X = df.drop('target', axis=1)
# Target (y): Posture label
y = df['target']

# Encode the text labels (e.g., 'Correct_Lordotic' -> 0, 'Wrong_Posture' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print(f"--- 🧠 Training Model on {len(X_train)} samples ---")

# n_estimators=100 provides a good balance between speed and precision
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Complete!")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- 📝 Classification Report ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE ASSETS FOR STREAMLIT
# ==========================================
MODEL_NAME = "lordotic_model.pkl"
ENCODER_NAME = "lordotic_label_encoder.pkl"

joblib.dump(model, MODEL_NAME)
joblib.dump(label_encoder, ENCODER_NAME)

print(f"\n🚀 SUCCESS: Saved '{MODEL_NAME}' and '{ENCODER_NAME}'")
print("You can now use these in your Chest Radiography Assistant app.")