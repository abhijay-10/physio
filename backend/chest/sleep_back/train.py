import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. LOAD DATASET (Headerless) ---
CSV_FILE = "sleep_back_dataset.csv"

print(f"--- 📂 Loading {CSV_FILE} ---")
try:
    # We use header=None because your CSV starts directly with numbers
    df = pd.read_csv(CSV_FILE, header=None)
    print(f"Total Samples Collected: {len(df)}")
except FileNotFoundError:
    print(f"❌ Error: {CSV_FILE} not found. Capture data first!")
    exit()

# --- 2. PREPROCESSING ---
# iloc[:, :-1] -> Takes columns 0 to 98 as Features (X)
# iloc[:, -1]  -> Takes column 99 as the Target (y)
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# Convert text labels (e.g., 'Correct_Sleep_Back') into numbers (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n--- 📊 Class Distribution ---")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"{u}: {c} samples")

# --- 3. TRAIN/TEST SPLIT ---
# Safety check: if you have very few samples, we disable 'stratify'
if np.min(np.bincount(y_encoded)) < 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- 4. TRAIN RANDOM FOREST ---
print(f"\n--- 🧠 Training Random Forest Model ---")
# n_estimators=100 is the sweet spot for coordinate-based pose detection
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. EVALUATION ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Complete!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# --- 6. SAVE FOR STREAMLIT ---
MODEL_NAME = "sleep_back_model.pkl"
ENCODER_NAME = "sleep_back_label_encoder.pkl"

joblib.dump(model, MODEL_NAME)
joblib.dump(label_encoder, ENCODER_NAME)

print(f"\n🚀 SUCCESS: Saved '{MODEL_NAME}' and '{ENCODER_NAME}'")