
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def calculate_metrics(y_true, predictions):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    modelling_table = pd.read_csv(
        DATA_DIR / "modelling_table.csv"
    )

    X = modelling_table[FEATURES].copy()
    y = modelling_table["is_delayed"].copy()

    required_columns = FEATURES + ["is_delayed"]

    if modelling_table[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in features or target. "
            "Check Part A before running Part C."
        )


 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )


    blind_predictions = np.zeros(
        len(y_val),
        dtype=int
    )

    model_rows = []

    blind_metrics = calculate_metrics(
        y_val,
        blind_predictions
    )

    model_rows.append({
        "model": "blind_all_on_time",
        **blind_metrics
    })


 
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_val)

    logistic_metrics = calculate_metrics(
        y_val,
        logistic_predictions
    )

    model_rows.append({
        "model": "logistic_regression",
        **logistic_metrics
    })


    tree_depths = [2, 3, 4, 5, 6, 8, 10]

    tree_results = []

    for depth in tree_depths:

        tree_model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=42
        )

        tree_model.fit(X_train, y_train)

        predictions = tree_model.predict(X_val)

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        tree_results.append({
            "depth": depth,
            "f1": metrics["f1"],
            "metrics": metrics
        })

    best_tree = min(
        tree_results,
        key=lambda result: (
            -result["f1"],
            result["depth"]
        )
    )

    best_tree_depth = best_tree["depth"]

    model_rows.append({
        "model": f"decision_tree_depth_{best_tree_depth}",
        **best_tree["metrics"]
    })



    default_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    default_forest.fit(X_train, y_train)

    forest_predictions = default_forest.predict(X_val)

    forest_metrics = calculate_metrics(
        y_val,
        forest_predictions
    )

    model_rows.append({
        "model": "random_forest_default",
        **forest_metrics
    })


    model_comparison = pd.DataFrame(
        model_rows,
        columns=[
            "model",
            "TN",
            "FP",
            "FN",
            "TP",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    model_comparison.to_csv(
        OUTPUT_DIR / "part_c_models.csv",
        index=False,
        float_format="%.4f"
    )



    forest_importances = default_forest.feature_importances_

    f_selector = SelectKBest(
        score_func=f_classif,
        k="all"
    )

    f_selector.fit(X_train, y_train)

    f_scores = f_selector.scores_

    feature_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": forest_importances
    })

    feature_scores["f_rank"] = (
        feature_scores["f_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    feature_scores["is_noise"] = (
        feature_scores["feature"].isin(NOISE_COLUMNS)
    )

    feature_scores = feature_scores.sort_values(
        ["f_rank", "feature"]
    ).reset_index(drop=True)

    feature_scores = feature_scores[
        [
            "feature",
            "f_score",
            "f_rank",
            "rf_importance",
            "is_noise"
        ]
    ]

    feature_scores.to_csv(
        OUTPUT_DIR / "part_c_feature_scores.csv",
        index=False,
        float_format="%.4f"
    )



    selectk_results = []

    for k in [5, 7, 10, 12, 17]:

        selectk_pipeline = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ])

        selectk_pipeline.fit(X_train, y_train)

        predictions = selectk_pipeline.predict(X_val)

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        selectk_results.append({
            "k": k,
            "val_f1": metrics["f1"]
        })

    selectk_df = pd.DataFrame(selectk_results)

    selectk_df.to_csv(
        OUTPUT_DIR / "part_c_selectk.csv",
        index=False,
        float_format="%.4f"
    )



    selector = SelectKBest(
        score_func=f_classif,
        k=7
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_val_selected = selector.transform(X_val)



    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid=param_grid,
        cv=5
    )

    grid_search.fit(
        X_train_selected,
        y_train
    )

    grid_predictions = grid_search.best_estimator_.predict(
        X_val_selected
    )

    grid_metrics = calculate_metrics(
        y_val,
        grid_predictions
    )



    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42
    )

    randomized_search.fit(
        X_train_selected,
        y_train
    )

    randomized_predictions = (
        randomized_search.best_estimator_.predict(
            X_val_selected
        )
    )

    randomized_metrics = calculate_metrics(
        y_val,
        randomized_predictions
    )



    default_forest_7 = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    default_forest_7.fit(
        X_train_selected,
        y_train
    )

    default_7_predictions = default_forest_7.predict(
        X_val_selected
    )

    default_7_metrics = calculate_metrics(
        y_val,
        default_7_predictions
    )



    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": grid_search.best_score_,
            "val_f1": grid_metrics["f1"]
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(randomized_search.best_params_),
            "best_cv_accuracy": randomized_search.best_score_,
            "val_f1": randomized_metrics["f1"]
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": str({
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_leaf": 1
            }),
            "best_cv_accuracy": np.nan,
            "val_f1": default_7_metrics["f1"]
        }
    ]

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df.to_csv(
        OUTPUT_DIR / "part_c_tuning.csv",
        index=False,
        float_format="%.4f"
    )



    tie_priority = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2
    }

    winning_row = min(
        tuning_rows,
        key=lambda row: (
            -row["val_f1"],
            tie_priority[row["search"]]
        )
    )

    chosen_from = winning_row["search"]

    if chosen_from == "grid_search":
        chosen_params = grid_search.best_params_

    elif chosen_from == "randomized_search":
        chosen_params = randomized_search.best_params_

    else:
        chosen_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }



    final_model = Pipeline([
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=7
            )
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **chosen_params
            )
        )
    ])

    final_model.fit(X_train, y_train)


    val_probabilities = final_model.predict_proba(
        X_val
    )[:, 1]

    thresholds = [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_val,
            predictions
        )

        threshold_results.append({
            "threshold": threshold,
            **metrics
        })

    threshold_df = pd.DataFrame(
        threshold_results,
        columns=[
            "threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "part_c_thresholds.csv",
        index=False,
        float_format="%.4f"
    )

 
    eligible_thresholds = threshold_df[
        threshold_df["recall"] >= 0.80
    ]

    if len(eligible_thresholds) > 0:
        chosen_threshold = float(
            eligible_thresholds["threshold"].max()
        )
    else:
        chosen_threshold = 0.1



    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= chosen_threshold
    ).astype(int)

    final_test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        **final_test_metrics
    }])

    final_test_df.to_csv(
        OUTPUT_DIR / "part_c_final_test.csv",
        index=False,
        float_format="%.4f"
    )



    max_depth_value = chosen_params["max_depth"]

    final_config = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": (
            "unlimited"
            if max_depth_value is None
            else max_depth_value
        ),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": chosen_from
    }])

    final_config.to_csv(
        OUTPUT_DIR / "part_c_final_config.csv",
        index=False,
        float_format="%.4f"
    )


if __name__ == "__main__":
    main()
