# Iteration 3 - Early Prediction of Higher Education Dropout
# Python implementation designed to correct the main issues identified in Iteration 2.
#
# Core design:
#   1) Binary operational target: Dropout vs Not Dropout
#   2) Strictly use enrolment + first-semester variables
#   3) Explicit assertion prevents all second-semester leakage
#   4) 60/20/20 stratified Train / Validation / Test
#   5) Algorithm comparison and hyperparameter tuning on TRAIN only
#   6) Classification-threshold tuning on VALIDATION only
#   7) TEST is touched once, after all choices are frozen
#   8) Broader EDA, feature construction, projection, and real merge with a codebook table
#   9) Honest success-criteria pass/fail table

# Recommended environment:
# pip install pandas numpy matplotlib scikit-learn xgboost ucimlrepo

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve, make_scorer
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------

def load_uci_data(local_csv=None):
    """
    Load UCI dataset 697.

    If local_csv is supplied, read that file.
    Otherwise use ucimlrepo.

    The original CSV normally uses ';' as separator.
    """
    if local_csv:
        local_csv = Path(local_csv)
        if not local_csv.exists():
            raise FileNotFoundError(local_csv)
        try:
            df = pd.read_csv(local_csv, sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(local_csv)
        except Exception:
            df = pd.read_csv(local_csv)
        return df

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Install ucimlrepo first: pip install ucimlrepo"
        ) from exc

    ds = fetch_ucirepo(id=697)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    return pd.concat([X, y], axis=1)


# Set to your downloaded data.csv path if you prefer local execution.
LOCAL_CSV = None
df_raw = load_uci_data(LOCAL_CSV)

print("Raw shape:", df_raw.shape)
print(df_raw.head())


# ---------------------------------------------------------------------
# 2. STANDARDISE FIELD NAMES
# ---------------------------------------------------------------------

def clean_column_name(col):
    col = str(col).strip().lower()
    col = col.replace("1st sem", "first sem").replace("2nd sem", "second sem")
    col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return col

df = df_raw.copy()
df.columns = [clean_column_name(c) for c in df.columns]

if "target" not in df.columns:
    raise ValueError("Target column was not found after column-name cleaning.")

print("\nCleaned columns:")
print(df.columns.tolist())


# ---------------------------------------------------------------------
# 3. DEFINE THE OPERATIONAL TARGET
# ---------------------------------------------------------------------
# Iteration 2 began as a 3-class problem but later switched to binary.
# Iteration 3 makes the relationship explicit:
#
#   Descriptive/original outcome: Dropout / Enrolled / Graduate
#   Operational intervention target: is_dropout = 1 if Dropout, otherwise 0
#
# This matches the business decision: "Who should receive early intervention?"

df["target_original"] = df["target"].astype(str).str.strip()
df["is_dropout"] = (df["target_original"].str.lower() == "dropout").astype(int)

print("\nOriginal target distribution:")
print(df["target_original"].value_counts(dropna=False))
print("\nBinary target distribution:")
print(df["is_dropout"].value_counts(dropna=False))
print(df["is_dropout"].value_counts(normalize=True).rename("proportion"))


# ---------------------------------------------------------------------
# 4. ACTUAL DATA INTEGRATION: MERGE COURSE CODEBOOK
# ---------------------------------------------------------------------
# This is a separate reference/codebook table based on the UCI metadata.
# It is merged into the student-level table using the Course code.
# The human-readable course_name is useful for EDA and reporting.
# We do NOT use course_name together with course code in the model because
# that would duplicate the same information.

course_lookup = pd.DataFrame({
    "course": [
        33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147,
        9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991
    ],
    "course_name": [
        "Biofuel Production Technologies",
        "Animation and Multimedia Design",
        "Social Service (evening attendance)",
        "Agronomy",
        "Communication Design",
        "Veterinary Nursing",
        "Informatics Engineering",
        "Equinculture",
        "Management",
        "Social Service",
        "Tourism",
        "Nursing",
        "Oral Hygiene",
        "Advertising and Marketing Management",
        "Journalism and Communication",
        "Basic Education",
        "Management (evening attendance)",
    ]
})

if "course" in df.columns:
    df = df.merge(course_lookup, on="course", how="left", validate="many_to_one")
    print("\nCourse-code integration check:")
    print(df[["course", "course_name"]].drop_duplicates().sort_values("course"))
    print("Unmatched course names:", df["course_name"].isna().sum())


# ---------------------------------------------------------------------
# 5. DATA QUALITY CHECKS
# ---------------------------------------------------------------------

quality_summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "n_unique": df.nunique(dropna=False)
}).sort_values(["missing", "n_unique"], ascending=[False, True])

print("\nData quality summary:")
print(quality_summary)

print("\nDuplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------------------
# 6. FEATURE CONSTRUCTION USING FIRST-SEMESTER DATA ONLY
# ---------------------------------------------------------------------

def safe_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce").fillna(0.0)
    den = pd.to_numeric(den, errors="coerce").fillna(0.0)
    return np.divide(
        num, den,
        out=np.zeros(len(num), dtype=float),
        where=den.to_numpy() != 0
    )

approved = "curricular_units_first_sem_approved"
enrolled = "curricular_units_first_sem_enrolled"
evaluations = "curricular_units_first_sem_evaluations"
without_eval = "curricular_units_first_sem_without_evaluations"

if approved in df.columns and enrolled in df.columns:
    df["first_sem_approval_ratio"] = safe_ratio(df[approved], df[enrolled])
    df["first_sem_unapproved_units"] = (
        pd.to_numeric(df[enrolled], errors="coerce").fillna(0)
        - pd.to_numeric(df[approved], errors="coerce").fillna(0)
    ).clip(lower=0)

if approved in df.columns and evaluations in df.columns:
    df["first_sem_approved_per_evaluation"] = safe_ratio(
        df[approved], df[evaluations]
    )

if without_eval in df.columns and enrolled in df.columns:
    df["first_sem_without_eval_ratio"] = safe_ratio(
        df[without_eval], df[enrolled]
    )

if {"debtor", "tuition_fees_up_to_date"}.issubset(df.columns):
    df["financial_risk_flag"] = (
        (pd.to_numeric(df["debtor"], errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(df["tuition_fees_up_to_date"], errors="coerce").fillna(1) == 0)
    ).astype(int)


# ---------------------------------------------------------------------
# 7. ENFORCE THE EARLY-WARNING FEATURE WINDOW
# ---------------------------------------------------------------------
# No second-semester field is allowed into the model.
# Protected/sensitive fields excluded by governance choice.

governance_exclusions = {
    "nacionality",
    "gender",
    "international",
    "educational_special_needs",
}

non_predictive_fields = {
    "target",
    "target_original",
    "is_dropout",
    "course_name",       # reporting-only label; course code remains available
}

second_semester_fields = {
    c for c in df.columns
    if ("second_sem" in c) or ("2nd_sem" in c) or ("2nd" in c and "sem" in c)
}

feature_cols = [
    c for c in df.columns
    if c not in governance_exclusions
    and c not in non_predictive_fields
    and c not in second_semester_fields
]

# Hard safety check: fail immediately if a late-stage variable slips through.
assert not any("second_sem" in c for c in feature_cols), feature_cols
assert not any("2nd" in c and "sem" in c for c in feature_cols), feature_cols

print("\nExcluded second-semester fields:")
print(sorted(second_semester_fields))

print("\nGovernance exclusions:")
print(sorted(governance_exclusions.intersection(df.columns)))

print("\nFinal early-warning feature count:", len(feature_cols))
print(feature_cols)

X = df[feature_cols].copy()
y = df["is_dropout"].copy()


# ---------------------------------------------------------------------
# 8. BROADER EDA
# ---------------------------------------------------------------------

def dropout_rate_by(column, top_n=None):
    temp = (
        df.groupby(column, dropna=False)["is_dropout"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "dropout_rate"})
          .sort_values("dropout_rate", ascending=False)
    )
    if top_n:
        temp = temp.head(top_n)
    print(f"\nDropout rate by {column}:")
    print(temp)
    return temp

# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
df["target_original"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Original outcome distribution")
ax.set_xlabel("Outcome")
ax.set_ylabel("Students")
plt.tight_layout()
plt.show()

# Important categorical/flag predictors
for col in [
    "tuition_fees_up_to_date",
    "debtor",
    "scholarship_holder",
    "daytime_evening_attendance",
]:
    if col in df.columns:
        rates = dropout_rate_by(col)
        fig, ax = plt.subplots(figsize=(6, 4))
        rates["dropout_rate"].plot(kind="bar", ax=ax)
        ax.set_title(f"Dropout rate by {col}")
        ax.set_ylabel("Dropout rate")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

# First-semester academic variables
for col in [
    "curricular_units_first_sem_approved",
    "curricular_units_first_sem_grade",
    "first_sem_approval_ratio",
    "first_sem_without_eval_ratio",
]:
    if col in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, group in df.groupby("is_dropout"):
            ax.hist(group[col].dropna(), bins=25, alpha=0.45, label=f"is_dropout={label}")
        ax.set_title(f"{col} by dropout status")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.show()

# Statistically justified age cohorts for EDA: quantile-based rather than arbitrary cutoffs
if "age_at_enrollment" in df.columns:
    try:
        df["age_quantile_group"] = pd.qcut(
            df["age_at_enrollment"],
            q=4,
            duplicates="drop"
        )
        dropout_rate_by("age_quantile_group")
    except Exception as e:
        print("Age qcut skipped:", e)

# Course-level exploration after the metadata merge
if "course_name" in df.columns:
    course_rates = (
        df.groupby("course_name")["is_dropout"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "dropout_rate"})
          .sort_values("dropout_rate", ascending=False)
    )
    print("\nDropout rate by course:")
    print(course_rates)


# ---------------------------------------------------------------------
# 9. TRAIN / VALIDATION / TEST DESIGN
# ---------------------------------------------------------------------
# 60% Train   -> algorithm comparison + hyperparameter tuning
# 20% Val     -> threshold selection only
# 20% Test    -> untouched until the very end

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,     # 25% of 80% = 20% of total
    stratify=y_dev,
    random_state=RANDOM_STATE,
)

print("\nSplit sizes:")
print("Train:", X_train.shape, y_train.value_counts().to_dict())
print("Valid:", X_val.shape, y_val.value_counts().to_dict())
print("Test :", X_test.shape, y_test.value_counts().to_dict())


# ---------------------------------------------------------------------
# 10. DEFINE CATEGORICAL VS NUMERIC FEATURES
# ---------------------------------------------------------------------

categorical_candidates = {
    "marital_status",
    "application_mode",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "displaced",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "financial_risk_flag",
}

categorical_cols = [c for c in X.columns if c in categorical_candidates]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("\nCategorical:", categorical_cols)
print("\nNumeric:", numeric_cols)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# Projection for linear model:
# Yeo-Johnson is a statistical power transformation that can handle zero
# and negative values. It is learned inside the pipeline on training folds only.
linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
            ]),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe()),
            ]),
            categorical_cols,
        ),
    ],
    remainder="drop",
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
            ]),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe()),
            ]),
            categorical_cols,
        ),
    ],
    remainder="drop",
)


# ---------------------------------------------------------------------
# 11. ALGORITHM COMPARISON ON TRAIN ONLY
# ---------------------------------------------------------------------

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())
scale_pos_weight = negative / max(positive, 1)
print("\nTraining scale_pos_weight:", scale_pos_weight)

models = {
    "LogisticRegression": Pipeline([
        ("prep", linear_preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
    "RandomForest": Pipeline([
        ("prep", tree_preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

try:
    from xgboost import XGBClassifier

    models["XGBoost"] = Pipeline([
        ("prep", tree_preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
except ImportError:
    XGBClassifier = None
    print("\nXGBoost not installed. Run: pip install xgboost")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
}

comparison_rows = []

for name, estimator in models.items():
    cv_result = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    row = {"model": name}
    for metric in scoring:
        row[f"cv_{metric}"] = cv_result[f"test_{metric}"].mean()
        row[f"cv_{metric}_std"] = cv_result[f"test_{metric}"].std()
    comparison_rows.append(row)

comparison = (
    pd.DataFrame(comparison_rows)
      .sort_values("cv_average_precision", ascending=False)
      .reset_index(drop=True)
)

print("\nTRAIN-only cross-validation comparison:")
print(comparison)


# ---------------------------------------------------------------------
# 12. HYPERPARAMETER TUNING ON TRAIN ONLY
# ---------------------------------------------------------------------

if XGBClassifier is not None:
    xgb_pipeline = Pipeline([
        ("prep", tree_preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    param_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.10],
        "model__subsample": [0.70, 0.85, 1.00],
        "model__colsample_bytree": [0.70, 0.85, 1.00],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1, 5, 10],
        "model__scale_pos_weight": [
            1.0,
            scale_pos_weight * 0.8,
            scale_pos_weight,
            scale_pos_weight * 1.2,
        ],
    }

    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("\nBest XGBoost parameters:")
    print(search.best_params_)
    print("Best TRAIN CV average precision:", search.best_score_)

else:
    # Fallback: use the strongest non-XGB model from TRAIN CV
    best_name = comparison.iloc[0]["model"]
    best_model = clone(models[best_name]).fit(X_train, y_train)
    print("\nFallback champion:", best_name)


# ---------------------------------------------------------------------
# 13. THRESHOLD SELECTION ON VALIDATION ONLY
# ---------------------------------------------------------------------
# We do not assume 0.50 is the best intervention threshold.
# Success targets:
#   Recall >= 0.85
#   Precision >= 0.70
#
# Rule:
#   1) If a threshold meets BOTH, choose the one with highest F2.
#   2) Else, among thresholds meeting recall >= 0.85, choose highest precision.
#   3) Else, choose threshold with highest F2.
#
# Crucially, NO test-set information is used here.

best_model.fit(X_train, y_train)
val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_rows = []
for threshold in np.arange(0.05, 0.951, 0.01):
    pred = (val_prob >= threshold).astype(int)
    threshold_rows.append({
        "threshold": threshold,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "f2": fbeta_score(y_val, pred, beta=2, zero_division=0),
        "accuracy": accuracy_score(y_val, pred),
    })

threshold_table = pd.DataFrame(threshold_rows)

both_ok = threshold_table[
    (threshold_table["recall"] >= 0.85)
    & (threshold_table["precision"] >= 0.70)
]

if not both_ok.empty:
    chosen = both_ok.sort_values(
        ["f2", "precision", "recall"],
        ascending=False
    ).iloc[0]
    threshold_reason = "Meets both recall and precision criteria"
else:
    recall_ok = threshold_table[threshold_table["recall"] >= 0.85]
    if not recall_ok.empty:
        chosen = recall_ok.sort_values(
            ["precision", "f2"],
            ascending=False
        ).iloc[0]
        threshold_reason = "No threshold met both criteria; maximize precision while preserving recall >= 0.85"
    else:
        chosen = threshold_table.sort_values("f2", ascending=False).iloc[0]
        threshold_reason = "Recall >= 0.85 was not achievable; maximize F2"

chosen_threshold = float(chosen["threshold"])

print("\nChosen validation threshold:", round(chosen_threshold, 3))
print("Reason:", threshold_reason)
print(chosen)


# Validation precision-recall tradeoff chart
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, val_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall_curve, precision_curve)
ax.axvline(0.85, linestyle="--", label="Recall target = 0.85")
ax.axhline(0.70, linestyle="--", label="Precision target = 0.70")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Validation Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 14. REFIT ON TRAIN + VALIDATION, THEN TOUCH TEST ONCE
# ---------------------------------------------------------------------

X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

final_model = clone(best_model)
final_model.fit(X_train_val, y_train_val)

test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= chosen_threshold).astype(int)

train_prob = final_model.predict_proba(X_train_val)[:, 1]
train_pred = (train_prob >= chosen_threshold).astype(int)


def lift_at_fraction(y_true, prob, fraction=0.20):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    n = len(y_true)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(prob)[::-1]
    top_rate = y_true[order[:k]].mean()
    base_rate = y_true.mean()
    return top_rate / base_rate if base_rate > 0 else np.nan


test_metrics = {
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall": recall_score(y_test, test_pred, zero_division=0),
    "f1": f1_score(y_test, test_pred, zero_division=0),
    "f2": fbeta_score(y_test, test_pred, beta=2, zero_division=0),
    "average_precision": average_precision_score(y_test, test_prob),
    "roc_auc": roc_auc_score(y_test, test_prob),
    "lift_at_20pct": lift_at_fraction(y_test, test_prob, 0.20),
}

train_accuracy = accuracy_score(y_train_val, train_pred)
test_accuracy = test_metrics["accuracy"]
stability_gap = abs(train_accuracy - test_accuracy)

print("\nFINAL TEST METRICS — test set used only here:")
for k, v in test_metrics.items():
    print(f"{k:20s}: {v:.4f}")

print(f"{'train_accuracy':20s}: {train_accuracy:.4f}")
print(f"{'stability_gap':20s}: {stability_gap:.4f}")


# Confusion matrix
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    display_labels=["Not Dropout", "Dropout"],
    cmap=None,
    ax=ax,
    values_format="d",
)
ax.set_title("Final Holdout Confusion Matrix")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 15. SUCCESS CRITERIA TABLE — HONEST PASS/FAIL
# ---------------------------------------------------------------------

success = pd.DataFrame([
    {
        "criterion": "Lift @ top 20%",
        "target": "> 2.5",
        "actual": test_metrics["lift_at_20pct"],
        "passed": test_metrics["lift_at_20pct"] > 2.5,
    },
    {
        "criterion": "Dropout recall",
        "target": ">= 0.85",
        "actual": test_metrics["recall"],
        "passed": test_metrics["recall"] >= 0.85,
    },
    {
        "criterion": "Dropout precision",
        "target": ">= 0.70",
        "actual": test_metrics["precision"],
        "passed": test_metrics["precision"] >= 0.70,
    },
    {
        "criterion": "Train-test accuracy gap",
        "target": "< 0.10",
        "actual": stability_gap,
        "passed": stability_gap < 0.10,
    },
])

print("\nSuccess criteria:")
print(success)

if success["passed"].all():
    print("\nConclusion: all predefined technical success criteria were met.")
else:
    print(
        "\nConclusion: at least one predefined criterion was not met. "
        "Do NOT claim the model is fully deployment-ready. "
        "Report the trade-off and recommend further iteration or pilot validation."
    )


# ---------------------------------------------------------------------
# 16. INTERPRETABLE COMPANION TREE
# ---------------------------------------------------------------------
# This model is not used to choose the champion after test-set evaluation.
# Its purpose is to provide transparent early-stage decision structure.

tree_model = Pipeline([
    ("prep", tree_preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

tree_model.fit(X_train_val, y_train_val)
tree_test_prob = tree_model.predict_proba(X_test)[:, 1]
tree_test_pred = (tree_test_prob >= 0.5).astype(int)

print("\nInterpretable companion tree holdout diagnostics:")
print("Accuracy :", accuracy_score(y_test, tree_test_pred))
print("Precision:", precision_score(y_test, tree_test_pred, zero_division=0))
print("Recall   :", recall_score(y_test, tree_test_pred, zero_division=0))


# ---------------------------------------------------------------------
# 17. FEATURE IMPORTANCE FOR FINAL XGBOOST (IF AVAILABLE)
# ---------------------------------------------------------------------

model_step = final_model.named_steps.get("model")
prep_step = final_model.named_steps.get("prep")

if hasattr(model_step, "feature_importances_"):
    feature_names = prep_step.get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model_step.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 final-model features:")
    print(importance.head(20))

    top = importance.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Final Model Feature Importance — Early Features Only")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # Safety proof for the report
    assert not importance["feature"].str.contains(
        "second_sem|2nd_sem", case=False, regex=True
    ).any()


# ---------------------------------------------------------------------
# 18. EXPORT TABLES FOR THE REPORT
# ---------------------------------------------------------------------

Path("iteration3_outputs").mkdir(exist_ok=True)

comparison.to_csv("iteration3_outputs/model_comparison_train_cv.csv", index=False)
threshold_table.to_csv("iteration3_outputs/validation_threshold_search.csv", index=False)
success.to_csv("iteration3_outputs/final_success_criteria.csv", index=False)

pd.DataFrame([{
    "chosen_threshold": chosen_threshold,
    "threshold_reason": threshold_reason,
    **test_metrics,
    "train_accuracy": train_accuracy,
    "stability_gap": stability_gap,
}]).to_csv("iteration3_outputs/final_holdout_metrics.csv", index=False)

print("\nSaved report tables to iteration3_outputs/")


# # ---------------------------------------------------------------------
# # 19. ITERATION 3 REPORTING CHECKLIST
# # ---------------------------------------------------------------------
# print("""
# ITERATION 3 REPORTING CHECKLIST
# -------------------------------
# [1] State that the operational model is binary: Dropout vs Not Dropout.
# [2] Explain that the original 3-class target is retained only for descriptive context.
# [3] Show the explicit exclusion of every second-semester variable.
# [4] Show broader raw-data EDA before modeling.
# [5] Document the course-code metadata merge as the integration step.
# [6] Document Yeo-Johnson power transformation for the linear baseline.
# [7] Explain 60/20/20 Train/Validation/Test and that Test remained untouched.
# [8] Compare algorithms using cross-validation on Train only.
# [9] Tune XGBoost on Train only.
# [10] Choose classification threshold using Validation only.
# [11] Evaluate Test exactly once.
# [12] Report all four predefined criteria as PASS/FAIL.
# [13] If precision < 0.70 or any criterion fails, state that the model is not yet fully deployment-ready.
# [14] Interpret only enrolment/first-semester predictors as actionable early-warning patterns.
# """)
