
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.shipment_prep import FEATURES


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, predictions):

    mse = mean_squared_error(y_true, predictions)

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, predictions)
    }



def main():

    modelling_table = pd.read_csv(
        DATA_DIR / "modelling_table.csv"
    )

    X = modelling_table[FEATURES].copy()
    y = modelling_table["delivery_hours"].copy()



    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )


    
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = np.abs(
        X_train_scaled.mean(axis=0)
    ).max()

    test_scaled_max_abs_mean = np.abs(
        X_test_scaled.mean(axis=0)
    ).max()

    checks = pd.DataFrame([
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean)
    ], columns=["metric", "value"])

    checks.to_csv(
        OUTPUT_DIR / "part_b_checks.csv",
        index=False,
        float_format="%.4f"
    )


    
    baseline_prediction = np.full(
        len(y_val),
        y_train.mean()
    )

    metrics_rows = []

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_prediction
    )

    metrics_rows.append({
        "model": "mean_baseline",
        "split": "val",
        **baseline_metrics
    })


   
    tree_depths = [1, 2, 3, 4, 6, 8, 12, None]

    tree_results = []
    tree_models = {}

    for depth in tree_depths:

        tree_model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        tree_model.fit(X_train, y_train)

        train_predictions = tree_model.predict(X_train)
        val_predictions = tree_model.predict(X_val)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_predictions)
        )

        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_predictions)
        )

        tree_results.append({
            "max_depth": (
                "unlimited" if depth is None else depth
            ),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        })

        tree_models[depth] = tree_model


    tree_depth_df = pd.DataFrame(tree_results)

    tree_depth_df.to_csv(
        OUTPUT_DIR / "part_b_tree_depth.csv",
        index=False,
        float_format="%.4f"
    )


    best_tree_depth = min(
        tree_depths,
        key=lambda depth: (
            next(
                row["val_rmse"]
                for row in tree_results
                if row["max_depth"] == (
                    "unlimited" if depth is None else depth
                )
            ),
            float("inf") if depth is None else depth
        )
    )



    candidate_models = {

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "decision_tree": DecisionTreeRegressor(
            max_depth=best_tree_depth,
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    fitted_models = {}
    validation_rmse = {}

    for model_name, model in candidate_models.items():

        model.fit(X_train, y_train)

        fitted_models[model_name] = model

        predictions = model.predict(X_val)

        model_metrics = calculate_metrics(
            y_val,
            predictions
        )

        validation_rmse[model_name] = model_metrics["RMSE"]

        metrics_rows.append({
            "model": model_name,
            "split": "val",
            **model_metrics
        })


   
    learning_rate_results = []

    for learning_rate in [0.01, 0.1, 1.0]:

        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=learning_rate,
            random_state=42
        )

        gb_model.fit(X_train, y_train)

        predictions = gb_model.predict(X_val)

        learning_rate_results.append({
            "learning_rate": learning_rate,
            "val_rmse": np.sqrt(
                mean_squared_error(y_val, predictions)
            ),
            "val_r2": r2_score(y_val, predictions)
        })

    learning_rate_df = pd.DataFrame(
        learning_rate_results
    )

    learning_rate_df.to_csv(
        OUTPUT_DIR / "part_b_gb_learning_rate.csv",
        index=False,
        float_format="%.4f"
    )


  
    ranked_models = sorted(
        validation_rmse.items(),
        key=lambda item: item[1]
    )

    top_two_models = [
        model_name
        for model_name, rmse in ranked_models[:2]
    ]


  
    X_train_val = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_val = pd.concat(
        [y_train, y_val],
        axis=0
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = []

    for model_name in top_two_models:

        cv_scores = cross_val_score(
            candidate_models[model_name],
            X_train_val,
            y_train_val,
            cv=kfold
        )

        cv_results.append({
            "model": model_name,
            "mean_r2": np.mean(cv_scores),
            "std_r2": np.std(cv_scores)
        })

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        OUTPUT_DIR / "part_b_cv.csv",
        index=False,
        float_format="%.4f"
    )


   
    final_model_name = ranked_models[0][0]

    final_model = fitted_models[final_model_name]

    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        **test_metrics
    })


   
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "model",
            "split",
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ]
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "part_b_metrics.csv",
        index=False,
        float_format="%.4f"
    )


if __name__ == "__main__":
    main()
