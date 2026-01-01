import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("parkinsons.csv").dropna()

# Select EXACTLY two features
features = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)"]
X = df[features]
y = df["status"]

# Train-test split (keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline: scaling + model
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "parkinsons_model.pkl")
