import os
import pickle
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =========================
# Load dataset
# =========================
path = kagglehub.dataset_download(
    "iabhishekofficial/mobile-price-classification"
)
csv_path = os.path.join(path, "train.csv")
df = pd.read_csv(csv_path)

X = df.drop("price_range", axis=1)
y = df["price_range"]

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Scale
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Models
# =========================
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss"
    )
}

# =========================
# Train & Evaluate
# =========================
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs, multi_class="ovr"),
        "Precision": precision_score(y_test, preds, average="weighted"),
        "Recall": recall_score(y_test, preds, average="weighted"),
        "F1": f1_score(y_test, preds, average="weighted"),
        "MCC": matthews_corrcoef(y_test, preds)
    }

# =========================
# SAVE IN SAME FOLDER
# =========================
with open("saved_models.pkl", "wb") as f:
    pickle.dump((models, results, scaler), f)

print("saved_models.pkl created INSIDE model/ folder only")
