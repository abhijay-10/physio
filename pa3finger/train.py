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
dataset_file = "pa_finger_dataset.csv"

try:
    df = pd.read_csv(dataset_file)
    print(f"✅ Dataset loaded successfully! Total samples: {len(df)}")
except FileNotFoundError:
    print(f"❌ Error: {dataset_file} not found. Please run the collector first.")
    exit()

# ==========================================
# 2. PREPROCESS DATA
# ==========================================
# X contains all 63 coordinates (x0, y0, z0 ... z20)
X = df.drop("label", axis=1)

# y contains the text labels ("PA Finger" or "Wrong")
y = df["label"]

# Convert text labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data: 80% for training, 20% for verifying accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. TRAIN THE CLASSIFIER
# ==========================================
print("Training PA Finger Model...")

# Using 100 trees for high precision
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"TRAINING ACCURACY: {accuracy * 100:.2f}%")
print("===================================\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE FILES
# ==========================================
# Using specific names to keep your models organized
model_name = "pa_finger_model.pkl"
encoder_name = "pa_finger_label_encoder.pkl"

joblib.dump(model, model_name)
joblib.dump(label_encoder, encoder_name)

print(f"✅ Model saved: {model_name}")
print(f"✅ Encoder saved: {encoder_name}")
print("\nYou are ready for the Streamlit test!")