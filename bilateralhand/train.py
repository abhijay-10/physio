import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD BILATERAL DATASET
# ==========================================
dataset_file = "bilateral_pa_dataset.csv"

try:
    df = pd.read_csv(dataset_file)
    print(f"✅ Bilateral Dataset loaded. Rows: {len(df)}")
except FileNotFoundError:
    print(f"❌ Error: {dataset_file} not found. Please collect data for two hands first.")
    exit()

# ==========================================
# 2. PREPROCESS DATA
# ==========================================
# Features (X): 126 columns (h0_x0...h1_z20)
X = df.drop("label", axis=1)

# Target (y): "Bilateral PA" or "Wrong"
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into Training (80%) and Testing (20%)
# Stratify ensures both classes are equally represented in split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 3. TRAIN RANDOM FOREST
# ==========================================
print("Training Bilateral Model (this may take a moment due to more features)...")

# We use 150 trees to handle the increased complexity of dual-hand data
model = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"BILATERAL MODEL ACCURACY: {accuracy * 100:.2f}%")
print("===================================\n")
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ==========================================
# 5. SAVE FILES
# ==========================================
model_filename = "bilateral_pa_model.pkl"
encoder_filename = "bilateral_label_encoder.pkl"

joblib.dump(model, model_filename)
joblib.dump(label_encoder, encoder_filename)

print(f"✅ Bilateral Model saved: {model_filename}")
print(f"✅ Bilateral Encoder saved: {encoder_filename}")