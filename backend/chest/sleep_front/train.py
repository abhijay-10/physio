import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. LOAD SUPINE DATASET
# ==========================================
DATASET_PATH = "sleep_front_dataset.csv" 

print(f"--- 📂 Loading {DATASET_PATH} ---")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: {DATASET_PATH} not found. Ensure you saved your data collection first.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Features (X): x0, y0, z0 ... x32, y32, z32 (99 total)
X = df.drop('target', axis=1)
# Target (y): 'Correct_Sleep_Front' or 'Wrong_Posture'
y = df['target']

# Convert text labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print(f"--- 🧠 Training Random Forest on {len(X_train)} samples ---")

# Standard Random Forest is very effective for single-frame posture detection
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Complete!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("\n--- 📝 Classification Report ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE ASSETS
# ==========================================
MODEL_NAME = "sleep_front_model.pkl"
ENCODER_NAME = "sleep_front_label_encoder.pkl"

joblib.dump(model, MODEL_NAME)
joblib.dump(label_encoder, ENCODER_NAME)

print(f"\n🚀 SUCCESS: Saved '{MODEL_NAME}' and '{ENCODER_NAME}'")