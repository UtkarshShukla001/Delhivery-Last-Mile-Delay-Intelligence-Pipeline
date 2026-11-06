
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import (
    FEATURES,
    clean_shipments
)



ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)



def main():

    modelling_table = pd.read_csv(
        DATA_DIR / "modelling_table.csv"
    )

    X = modelling_table[FEATURES].copy()
    y = modelling_table["is_delayed"].copy()

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


 
    config = pd.read_csv(
        OUTPUT_DIR / "part_c_final_config.csv"
    ).iloc[0]

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    min_samples_leaf = int(config["min_samples_leaf"])

    depth_value = config["max_depth"]

    if pd.isna(depth_value) or str(depth_value).lower() == "unlimited":
        max_depth = None
    else:
        max_depth = int(float(depth_value))

    threshold = float(config["threshold"])


 
    final_model = Pipeline([
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=selector_k
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        )
    ])

    final_model.fit(X_train, y_train)



    joblib.dump(
        final_model,
        MODEL_DIR / "delay_model.pkl"
    )



    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv(
        MODEL_DIR / "delay_threshold.csv",
        index=False,
        float_format="%.4f"
    )



    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        MODEL_DIR / "feature_order.csv",
        index=False
    )



    raw_shipments = pd.read_csv(
        DATA_DIR / "raw_shipments.csv"
    )

    clean_shipment, medians = clean_shipments(
        raw_shipments
    )

    impute_values = pd.DataFrame({
        "column": list(medians.keys()),
        "median": list(medians.values())
    })

    impute_values.to_csv(
        MODEL_DIR / "impute_values.csv",
        index=False,
        float_format="%.4f"
    )



    reloaded_model = joblib.load(
        MODEL_DIR / "delay_model.pkl"
    )

    original_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    reloaded_probabilities = reloaded_model.predict_proba(
        X_test
    )[:, 1]

    reload_max_abs_probability_diff = np.max(
        np.abs(
            original_probabilities -
            reloaded_probabilities
        )
    )



    reload_check = pd.DataFrame([
        ("test_rows_scored", len(X_test)),
        (
            "reload_max_abs_probability_diff",
            reload_max_abs_probability_diff
        ),
        ("threshold", threshold)
    ], columns=["metric", "value"])

    reload_check.to_csv(
        OUTPUT_DIR / "part_d_reload_check.csv",
        index=False,
        float_format="%.4f"
    )


if __name__ == "__main__":
    main()
