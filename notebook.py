import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("personal_finance_advisor_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Categorical Columns
cat_cols = [
    "existing_loan",
    "investment_experience",
    "risk_tolerance",
    "employment_type",
    "investment_goal"
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target Encoding
target_encoder = LabelEncoder()
df["recommendation"] = target_encoder.fit_transform(df["recommendation"])

# Features and Target
X = df.drop("recommendation", axis=1)
y = df["recommendation"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)





from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))



from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

print("Precision:", precision_score(y_test, pred, average="weighted"))
print("Recall:", recall_score(y_test, pred, average="weighted"))
print("F1 Score:", f1_score(y_test, pred, average="weighted"))




import joblib

joblib.dump(model, "finance_model.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")