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
dataset_file = "oblique_thumb_dataset.csv"

try:
    df = pd.read_csv(dataset_file)
    print(f"✅ Dataset loaded. Found {len(df)} rows.")
except FileNotFoundError:
    print("❌ CSV file not found. Make sure the filename matches your collector output.")
    exit()

# ==========================================
# 2. PREPROCESS DATA
# ==========================================
# Features (X): x0, y0, z0 ... z20
X = df.drop("label", axis=1)

# Target (y): "Oblique Thumb" or "Wrong"
y = df["label"]

# Encode text labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("Training Oblique Thumb Model...")

# Random Forest is highly effective for landmark-based classification
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("===================================\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE FILES
# ==========================================
# We save these with specific names to avoid overwriting your other models
model_name = "oblique_thumb_model.pkl"
encoder_name = "oblique_label_encoder.pkl"

joblib.dump(model, model_name)
joblib.dump(label_encoder, encoder_name)

print(f"✅ Model saved as: {model_name}")
print(f"✅ Encoder saved as: {encoder_name}")