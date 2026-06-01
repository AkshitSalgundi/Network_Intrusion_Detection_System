"""
train_model.py
Trains the Random Forest model and saves everything needed by the Streamlit app.
Run this once before launching app.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("archive/UNSW_NB15_training-set.csv")
test  = pd.read_csv("archive/UNSW_NB15_testing-set.csv")
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── 2. Drop useless columns ─────────────────────────────────────────────────
df_train = train.drop(columns=["id", "attack_cat"])
df_test  = test.drop(columns=["id", "attack_cat"])

# ── 3. Save unique values for dropdowns in the Streamlit app ────────────────
cat_cols    = ["proto", "service", "state"]
unique_vals = {col: sorted(df_train[col].unique().tolist()) for col in cat_cols}

# ── 4. Encode text columns to numbers ───────────────────────────────────────
text_cols = df_train.select_dtypes(include="str").columns.tolist()
encoders  = {}

for col in text_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col].astype(str))
    known = list(le.classes_)
    df_test[col]  = df_test[col].astype(str).apply(
        lambda x: int(le.transform([x])[0]) if x in known else -1
    )
    encoders[col] = le

# ── 5. Split features and target ────────────────────────────────────────────
X_train = df_train.drop(columns=["label"])
y_train = df_train["label"]
X_test  = df_test.drop(columns=["label"])
y_test  = df_test["label"]

# ── 6. Feature engineering (done BEFORE scaling) ────────────────────────────
def add_features(df):
    df = df.copy()
    df["total_bytes"]   = df["sbytes"] + df["dbytes"]
    df["total_pkts"]    = df["spkts"]  + df["dpkts"]
    df["bytes_per_pkt"] = df["total_bytes"] / (df["total_pkts"] + 1)
    df["ttl_diff"]      = (df["sttl"] - df["dttl"]).abs()
    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

# save medians as a plain dict to avoid pandas version mismatch when loading
medians = X_train.median().to_dict()

# ── 7. Scale ────────────────────────────────────────────────────────────────
scaler      = StandardScaler()
X_train_s   = scaler.fit_transform(X_train)
X_test_s    = scaler.transform(X_test)

# ── 8. Train ────────────────────────────────────────────────────────────────
print("Training Random Forest (100 trees) ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
print("Done!")

# ── 9. Evaluate ─────────────────────────────────────────────────────────────
preds = rf.predict(X_test_s)
print(f"\nTest Accuracy: {accuracy_score(y_test, preds)*100:.2f}%\n")
print(classification_report(y_test, preds, target_names=["Normal", "Attack"]))

# pre-compute dashboard data so the app loads instantly
report_dict  = classification_report(y_test, preds, target_names=["Normal", "Attack"], output_dict=True)
conf_matrix  = confusion_matrix(y_test, preds).tolist()

# save as plain dict — avoids pandas version mismatch when loading in the app
pairs    = sorted(zip(rf.feature_importances_.tolist(), X_train.columns.tolist()), reverse=True)[:15]
feat_imp = {
    "feature":    [p[1] for p in pairs],
    "importance": [p[0] for p in pairs],
}

attack_dist  = train["attack_cat"].value_counts().to_dict()
label_dist   = train["label"].value_counts().to_dict()

dataset_info = {
    "train_rows":  len(train),
    "test_rows":   len(test),
    "n_features":  len(X_train.columns),
    "n_attacks":   int((train["label"] == 1).sum()),
    "n_normal":    int((train["label"] == 0).sum()),
}

# ── 10. Save everything ─────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(rf,                       "model/rf_model.pkl")
joblib.dump(scaler,                   "model/scaler.pkl")
joblib.dump(encoders,                 "model/encoders.pkl")
joblib.dump(list(X_train.columns),    "model/feature_names.pkl")
joblib.dump(medians,                  "model/medians.pkl")
joblib.dump(unique_vals,              "model/unique_vals.pkl")
joblib.dump(report_dict,              "model/report.pkl")
joblib.dump(conf_matrix,              "model/conf_matrix.pkl")
joblib.dump(feat_imp,                 "model/feat_importance.pkl")
joblib.dump(attack_dist,              "model/attack_dist.pkl")
joblib.dump(dataset_info,             "model/dataset_info.pkl")

print("\nAll model files saved to  model/")
print("Feature count:", len(X_train.columns))
print("\nYou can now run:  streamlit run app.py")
