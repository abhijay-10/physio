import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. LOAD DATASET ---
DATA_PATH = "chest_ap_stable_dataset.csv"

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Please collect data first!")
    exit()

# Read the CSV (Assuming no header)
df = pd.read_csv(DATA_PATH, header=None)
print(f"✅ Loaded {len(df)} total samples.")

# --- 2. DATA CLEANING (The Fix) ---
# Separate features and labels
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].astype(str).values # Convert to string to handle '99' vs 'Correct'

# Count occurrences of each class
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"📊 Initial Class Distribution: {class_counts}")

# FILTER: Keep only classes with at least 5 samples to ensure split stability
MIN_SAMPLES = 5
valid_classes = [cls for cls, count in class_counts.items() if count >= MIN_SAMPLES]
mask = np.isin(y, valid_classes)

X_clean = X[mask]
y_clean = y[mask]

print(f"🧹 Cleaned Dataset: {len(X_clean)} samples remaining.")
print(f"🚫 Removed classes: {[cls for cls in unique if cls not in valid_classes]}")

# --- 3. ENCODING ---
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)
class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
print(f"🎯 Target Classes: {class_mapping}")

# --- 4. TRAIN/TEST SPLIT ---
# Now stratify will work because all classes have > 1 member
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# --- 5. RANDOM FOREST TRAINING ---
print("🚀 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=20, 
    n_jobs=-1, 
    random_state=42
)

rf_model.fit(X_train, y_train)

# --- 6. EVALUATION ---
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"🏆 MODEL ACCURACY: {acc * 100:.2f}%")
print("="*30)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 7. SAVE ASSETS ---
joblib.dump(rf_model, "chest_ap_rf_model.pkl")
joblib.dump(le, "chest_ap_label_encoder.pkl")

print("\n📦 Assets saved: chest_ap_rf_model.pkl & chest_ap_label_encoder.pkl")