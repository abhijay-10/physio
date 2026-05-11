import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# LOAD DATASET
# ==========================================
dataset_file = "hand_dataset.csv"

df = pd.read_csv(dataset_file)

# ==========================================
# FEATURES AND LABELS
# ==========================================
X = df.drop("label", axis=1)

y = df["label"]

# ==========================================
# ENCODE LABELS
# ==========================================
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# TEST ACCURACY
# ==========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("===================================\n")

# ==========================================
# SAVE MODEL
# ==========================================
joblib.dump(
    model,
    "hand_model.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("✅ hand_model.pkl saved")
print("✅ label_encoder.pkl saved")
print("✅ Training complete")