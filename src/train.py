from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_score, recall_score
)
import joblib
import json
import os
from preprocess import load_and_preprocess


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  Model: {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "name": name,
        "model": model,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "TN": int(cm[0][0]),
            "FP": int(cm[0][1]),
            "FN": int(cm[1][0]),
            "TP": int(cm[1][1]),
        },
    }


def train_model():
    X, y, _ = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ("Logistic Regression", LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
    ]

    results = []
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n{'*'*45}")
    print(f"  Best model: {best['name']} (Accuracy: {best['accuracy']:.4f})")
    print(f"{'*'*45}")
    joblib.dump(best["model"], "models/best_model.pkl")
    print(f"  Saved to models/best_model.pkl")

    # Save metrics for the API/dashboard
    metrics_payload = {
        "best_model": best["name"],
        "models": [
            {k: v for k, v in r.items() if k != "model"}
            for r in results
        ],
    }
    os.makedirs("models", exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("  Metrics saved to models/metrics.json")

    return best["model"]


if __name__ == "__main__":
    train_model()
