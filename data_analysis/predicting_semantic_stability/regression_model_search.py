"""Grouped model search helpers for semantic-drift regression notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONSTANT_DROP_FEATURES = [
    "word_token_count",
    "description_n",
    "golden_description_embedding_n",
    "word_embedding_norm",
]

REDUNDANT_DROP_FEATURES = [
    "is_high_frequency_category",
    "is_concrete_category",
    "FREQcount",
    "SUBTLWF",
    "CDcount",
    "SUBTLCD",
    "Lg10CD",
    "FreqCount",
    "CD_count",
    "CD",
    "category_centroid_distance",
    "parameter_b",
]

ALL_CANDIDATE_FEATURES = [
    "category",
    "is_high_frequency_category",
    "is_concrete_category",
    "word_length_chars",
    "word_token_count",
    "FREQcount",
    "CDcount",
    "SUBTLWF",
    "Lg10WF",
    "SUBTLCD",
    "Lg10CD",
    "FreqCount",
    "LogFreq(Zipf)",
    "CD_count",
    "CD",
    "DomPoS",
    "word_embedding_norm",
    "neighbor_density_top10",
    "nearest_neighbor_similarity",
    "all_words_centroid_similarity",
    "category_centroid_similarity",
    "category_centroid_distance",
    "description_n",
    "mean_description_chars",
    "std_description_chars",
    "mean_description_tokens",
    "std_description_tokens",
    "mean_description_ttr",
    "corpus_description_ttr",
    "golden_description_embedding_n",
    "mean_pairwise_golden_description_similarity",
    "mean_word_to_golden_description_similarity",
    "model_family",
    "parameter_b",
    "log_parameter_b",
    "pipeline",
]

SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


@dataclass(frozen=True)
class SearchSettings:
    n_group_splits: int = 10
    group_test_size: float = 0.20
    n_random_search_iter: int = 24
    random_state: int = 42


def build_feature_sets(neighbor_k: int = 10) -> dict[str, list[str]]:
    neighbor_density = f"neighbor_density_top{neighbor_k}"
    pruned_main = [
        "category",
        "pipeline",
        "model_family",
        "log_parameter_b",
        "word_length_chars",
        "Lg10WF",
        "LogFreq(Zipf)",
        "DomPoS",
        neighbor_density,
        "nearest_neighbor_similarity",
        "all_words_centroid_similarity",
        "category_centroid_similarity",
        "mean_description_tokens",
        "std_description_tokens",
        "mean_description_ttr",
        "mean_pairwise_golden_description_similarity",
        "mean_word_to_golden_description_similarity",
    ]

    pruned_binary_category = [
        feature for feature in pruned_main if feature != "category"
    ] + [
        "is_high_frequency_category",
        "is_concrete_category",
    ]

    word_description_only = [
        "category",
        "word_length_chars",
        "Lg10WF",
        "LogFreq(Zipf)",
        "DomPoS",
        neighbor_density,
        "nearest_neighbor_similarity",
        "all_words_centroid_similarity",
        "category_centroid_similarity",
        "mean_description_tokens",
        "std_description_tokens",
        "mean_description_ttr",
        "mean_pairwise_golden_description_similarity",
        "mean_word_to_golden_description_similarity",
    ]

    return {
        "pruned_main": pruned_main,
        "pruned_binary_category": pruned_binary_category,
        "model_pipeline_only": ["pipeline", "model_family", "log_parameter_b"],
        "word_description_only": word_description_only,
    }


def feature_diagnostics(
    data: pd.DataFrame,
    candidate_features: list[str],
    selected_features: list[str],
) -> pd.DataFrame:
    rows = []
    selected = set(selected_features)
    for feature in [col for col in candidate_features if col in data.columns]:
        series = data[feature]
        numeric = pd.api.types.is_numeric_dtype(series)
        numeric_values = pd.to_numeric(series, errors="coerce") if numeric else None
        reason = "kept"
        if feature in CONSTANT_DROP_FEATURES:
            reason = "drop_constant_or_near_constant"
        elif feature in REDUNDANT_DROP_FEATURES:
            reason = "drop_redundant_or_less_stable_encoding"
        elif feature not in selected:
            reason = "not_in_selected_feature_set"

        rows.append(
            {
                "feature": feature,
                "selected": feature in selected,
                "drop_reason": reason,
                "dtype": str(series.dtype),
                "nunique": int(series.nunique(dropna=True)),
                "missing_frac": float(series.isna().mean()),
                "std_if_numeric": (
                    float(numeric_values.std()) if numeric_values is not None else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["selected", "drop_reason", "feature"],
        ascending=[False, True, True],
    )


def clean_model_frame(
    data: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame]:
    base = data.dropna(subset=[target]).copy().replace({pd.NA: np.nan})
    features = [feature for feature in features if feature in base.columns]

    numeric_cols = []
    categorical_cols = []
    dropped = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(base[feature]):
            base[feature] = pd.to_numeric(base[feature], errors="coerce")
            if base[feature].notna().sum() == 0 or base[feature].nunique(dropna=True) <= 1:
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "all_missing_or_no_variance_after_cleaning",
                    }
                )
            else:
                numeric_cols.append(feature)
        else:
            base[feature] = base[feature].astype("object").where(base[feature].notna(), np.nan)
            base[feature] = base[feature].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
            if base[feature].nunique(dropna=True) <= 1:
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "categorical_no_variance_after_cleaning",
                    }
                )
            else:
                categorical_cols.append(feature)

    cleaned_features = numeric_cols + categorical_cols
    x = base[cleaned_features].copy()
    y = pd.to_numeric(base[target], errors="coerce")
    groups = base["word_norm"].astype(str)
    return base, x, y, groups, numeric_cols, categorical_cols, pd.DataFrame(dropped)


def make_preprocess(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_model_specs(settings: SearchSettings) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "mean_dummy": {
            "estimator": DummyRegressor(strategy="mean"),
            "search": "none",
            "params": {},
        },
        "linear_regression": {
            "estimator": LinearRegression(),
            "search": "none",
            "params": {},
        },
        "ridge": {
            "estimator": Ridge(random_state=settings.random_state),
            "search": "grid",
            "params": {
                "model__alpha": np.logspace(-4, 4, 17),
            },
        },
        "random_forest": {
            "estimator": RandomForestRegressor(
                random_state=settings.random_state,
                n_jobs=-1,
            ),
            "search": "random",
            "n_iter": settings.n_random_search_iter,
            "params": {
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 3, 6, 10],
                "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
            },
        },
    }

    try:
        from xgboost import XGBRegressor

        specs["xgboost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=settings.random_state,
                n_jobs=-1,
            ),
            "search": "random",
            "n_iter": settings.n_random_search_iter,
            "params": {
                "model__n_estimators": [250, 400, 650, 900],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.01, 0.03, 0.06, 0.10],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.75, 0.9, 1.0],
                "model__min_child_weight": [1, 3, 6],
                "model__reg_lambda": [0.5, 1.0, 3.0, 8.0],
            },
        }
    except ImportError:
        pass

    return specs


def _summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
    }


def _summarize_search(search: GridSearchCV | RandomizedSearchCV) -> dict[str, float]:
    best_idx = search.best_index_
    return {
        "r2_mean": float(search.cv_results_["mean_test_r2"][best_idx]),
        "r2_std": float(search.cv_results_["std_test_r2"][best_idx]),
        "mae_mean": float(-search.cv_results_["mean_test_mae"][best_idx]),
        "mae_std": float(search.cv_results_["std_test_mae"][best_idx]),
        "rmse_mean": float(-search.cv_results_["mean_test_rmse"][best_idx]),
        "rmse_std": float(search.cv_results_["std_test_rmse"][best_idx]),
    }


def run_grouped_model_search(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    feature_set_name: str = "pruned_main",
    settings: SearchSettings | None = None,
    progress: Any = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, dict[str, Any]]:
    settings = settings or SearchSettings()
    _, x, y, groups, numeric_cols, categorical_cols, dropped = clean_model_frame(
        data,
        target,
        features,
    )
    preprocess = make_preprocess(numeric_cols, categorical_cols)
    cv = GroupShuffleSplit(
        n_splits=settings.n_group_splits,
        test_size=settings.group_test_size,
        random_state=settings.random_state,
    )
    specs = build_model_specs(settings)

    rows = []
    fitted = {}
    model_items = list(specs.items())
    if progress is not None:
        model_items = progress(model_items, desc=f"Grouped search for {target}", unit="model")

    for model_name, spec in model_items:
        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", spec["estimator"]),
            ]
        )

        if spec["search"] == "none":
            scores = cross_validate(
                pipe,
                x,
                y,
                groups=groups,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1,
                return_estimator=False,
            )
            cv_result = _summarize_scores(scores)
            pipe.fit(x, y)
            fitted[model_name] = pipe
            best_params: dict[str, Any] = {}
        else:
            search_cls = GridSearchCV if spec["search"] == "grid" else RandomizedSearchCV
            search_kwargs: dict[str, Any] = {
                "estimator": pipe,
                "scoring": SCORING,
                "refit": "rmse",
                "cv": cv,
                "n_jobs": -1,
                "return_train_score": False,
            }
            if spec["search"] == "grid":
                search_kwargs["param_grid"] = spec["params"]
            else:
                search_kwargs["param_distributions"] = spec["params"]
                search_kwargs["n_iter"] = spec.get("n_iter", settings.n_random_search_iter)
                search_kwargs["random_state"] = settings.random_state

            search = search_cls(**search_kwargs)
            search.fit(x, y, groups=groups)
            cv_result = _summarize_search(search)
            fitted[model_name] = search.best_estimator_
            best_params = search.best_params_

        rows.append(
            {
                "model": model_name,
                "target": target,
                "feature_set": feature_set_name,
                "search": spec["search"],
                "cv_r2_mean": cv_result["r2_mean"],
                "cv_r2_std": cv_result["r2_std"],
                "cv_mae_mean": cv_result["mae_mean"],
                "cv_mae_std": cv_result["mae_std"],
                "cv_rmse_mean": cv_result["rmse_mean"],
                "cv_rmse_std": cv_result["rmse_std"],
                "best_params": best_params,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)
    info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_rows": len(x),
        "n_groups": int(groups.nunique()),
    }
    return metrics, fitted, dropped, info
