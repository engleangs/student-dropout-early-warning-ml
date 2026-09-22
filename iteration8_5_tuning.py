"""Optional hyperparameter exploration for the next modelling iteration.

This module is deliberately separate from ``iteration3.py``. Importing it
does not fit a model or change the current Iteration 3 workflow. Call the
builder or runner functions explicitly in a future experiment, using only the
training split for model selection.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def get_xgboost_param_grid() -> dict[str, list[Any]]:
    """Return a fresh copy of the proposed XGBoost search space."""
    return {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__gamma": [0, 0.1, 0.3, 0.5],
        "model__subsample": [0.65, 0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.65, 0.75, 0.85, 1.0],
        "model__reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "model__reg_lambda": [1, 3, 5, 10],
        # XGBoost uses scale_pos_weight (not scale_post_weight).
        "model__scale_pos_weight": [1.0, 1.2, 1.4, 1.6, 1.8],
    }


def get_random_forest_param_grid() -> dict[str, list[Any]]:
    """Return a fresh copy of the proposed Random Forest search space."""
    return {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [5, 8, 10, 15, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [2, 4, 8, 12],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [
            "balanced",
            "balanced_subsample",
            {0: 1, 1: 1.2},
            {0: 1, 1: 1.4},
            {0: 1, 1: 1.6},
        ],
    }


def build_xgboost_search(
    preprocessor,
    cv,
    *,
    n_iter: int = 60,
    scoring: str = "average_precision",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """Create an unfitted XGBoost search without changing current models."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost exploration requires: pip install xgboost"
        ) from exc

    pipeline = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )),
    ])

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_xgboost_param_grid(),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )


def build_random_forest_search(
    preprocessor,
    cv,
    *,
    n_iter: int = 50,
    scoring: str = "average_precision",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """Create an unfitted Random Forest search without changing current models."""
    pipeline = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
        )),
    ])

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_random_forest_param_grid(),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )


def run_hyperparameter_exploration(
    X_train,
    y_train,
    preprocessor,
    cv,
    *,
    xgboost_n_iter: int = 60,
    random_forest_n_iter: int = 50,
) -> dict[str, RandomizedSearchCV]:
    """Fit both optional searches on training data and return them by name.

    This function must not receive validation or test data. It is not called
    by the current Iteration 3 workflow.
    """
    searches = {
        "RandomForest": build_random_forest_search(
            preprocessor,
            cv,
            n_iter=random_forest_n_iter,
        ),
        "XGBoost": build_xgboost_search(
            preprocessor,
            cv,
            n_iter=xgboost_n_iter,
        ),
    }

    for search in searches.values():
        search.fit(X_train, y_train)

    return searches


def summarize_searches(
    searches: dict[str, RandomizedSearchCV],
) -> pd.DataFrame:
    """Summarize fitted searches without selecting on validation/test data."""
    rows = []
    for model_name, search in searches.items():
        if not hasattr(search, "best_score_"):
            raise ValueError(f"{model_name} search has not been fitted.")
        rows.append({
            "model": model_name,
            "best_cv_average_precision": search.best_score_,
            "best_params": search.best_params_,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("best_cv_average_precision", ascending=False)
        .reset_index(drop=True)
    )


def export_search_results(
    searches: dict[str, RandomizedSearchCV],
    output_dir: str | Path,
) -> list[Path]:
    """Export fitted CV results when the future exploration is run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    file_stems = {
        "RandomForest": "random_forest",
        "XGBoost": "xgboost",
    }
    for model_name, search in searches.items():
        if not hasattr(search, "cv_results_"):
            raise ValueError(f"{model_name} search has not been fitted.")
        file_stem = file_stems.get(model_name, model_name.lower())
        output_path = output_dir / f"{file_stem}_randomized_search_results.csv"
        pd.DataFrame(search.cv_results_).to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths
