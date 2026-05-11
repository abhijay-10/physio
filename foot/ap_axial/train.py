import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. LOAD YOUR CSV DATASET
# ==========================================
DATASET_PATH = "foot_radiography_dataset.csv" # Ensure this matches your filename

print("--- 📂 Loading Foot Dataset ---")
df = pd.read_csv(DATASET_PATH)

# ==========================================
# 2. PREPROCESSING
# ==========================================
# X contains all columns except 'target' (x0...z32)
X = df.drop('target', axis=1)
# y contains the clinical labels (e.g., AP_Axial_Foot)
y = df['target']

# Convert text labels into numbers (e.g., AP_Axial_Foot -> 0, Wrong -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data: 80% for training, 20% for verifying accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print(f"--- 🧠 Training Model on {len(X_train)} samples ---")

# We use 100 trees (n_estimators) for high clinical precision
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Finished!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("\n--- 📝 Classification Report ---")
# This shows how well the model identifies each specific posture
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. EXPORT FOR STREAMLIT APP
# ==========================================
MODEL_FILE = "foot_model.pkl"
ENCODER_FILE = "foot_label_encoder.pkl"

joblib.dump(clf, MODEL_FILE)
joblib.dump(label_encoder, ENCODER_FILE)

print(f"\n🚀 SUCCESS: Models saved as '{MODEL_FILE}' and '{ENCODER_FILE}'")