import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# Load JSON data
with open("dataset/asl_data.json", "r") as f:
    data = json.load(f)

X = []
y = []

for sample in data:
    X.append(sample["landmarks"])
    y.append(sample["label"])

X = np.array(X)
y = np.array(y)

# Encode labels (e.g., "A" → 0, "B" → 1, etc.)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and label encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/asl_knn_model.joblib")
joblib.dump(encoder, "model/label_encoder.joblib")

print("[✅] Model and label encoder saved to 'model/' folder.")
