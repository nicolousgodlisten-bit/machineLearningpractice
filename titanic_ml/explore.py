from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Pakia data (tunatumia pathlib ili njia isivunjike)
BASE_DIR = Path(__file__).resolve().parent
train_df = pd.read_csv(BASE_DIR / "train.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")

print("=== 1. UKAGUZI WA MWANZO ===")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nMissing values kwenye Train:\n", train_df.isnull().sum())
print("\nMissing values kwenye Test:\n", test_df.isnull().sum())

# 2. Chagua safu (features) za kuanzia
# Tunaacha majina/tiketi/cabin za mwanzo ili tuwe na baseline rahisi na safi
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target_col = "Survived"

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# 3. Safisha na badilisha data ziwe namba (Data Cleaning & Preprocessing)
# Muhimu: tunatumia median na mode kutoka kwenye X (train) ili kuepuka data leakage
age_median = X["Age"].median()
fare_median = X["Fare"].median()
embarked_mode = X["Embarked"].mode()[0]

for df in (X, X_test):
    # Jaza mapengo (missing values)
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # Badilisha maandishi kuwa namba
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("\nMissing values zilizobaki kwenye X:", X.isnull().sum().sum())
print("Missing values zilizobaki kwenye X_test:", X_test.isnull().sum().sum())

# 4. Gawanya data ya train katika Training na Validation set
# Hatupimi accuracy kwenye test.csv moja kwa moja kwa sababu haina safu ya 'Survived'
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # huweka uwiano sawa wa waliookoka/wasiookoka kwenye train na val
)

# 5. Jenga na fundisha model ya kwanza (Baseline Model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Pima ubora wa model kwenye Validation set
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)

print("\n=== 2. MATOKEO YA MODEL (VALIDATION) ===")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
print("\nClassification Report:\n", classification_report(y_val, val_preds))

# Angalia feature gani zimechangia zaidi maamuzi ya model
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("Feature Importances:\n", importances)

# 7. Tengeneza utabiri wa mwisho kwa ajili ya test.csv na hifadhi CSV
test_preds = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_preds,
})
submission_path = BASE_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nUtabiri umehifadhiwa kwenye: {submission_path}")