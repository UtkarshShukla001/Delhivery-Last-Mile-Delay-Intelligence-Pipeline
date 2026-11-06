
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from src.shipment_prep import (
    clean_shipments,
    build_features
)



ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"



def main():

    new_raw = pd.read_csv(
        DATA_DIR / "new_shipments.csv"
    )

    hubs = pd.read_csv(
        DATA_DIR / "hubs.csv"
    )



    model = joblib.load(
        MODEL_DIR / "delay_model.pkl"
    )

    threshold = float(
        pd.read_csv(
            MODEL_DIR / "delay_threshold.csv"
        )["threshold"].iloc[0]
    )

    feature_order = (
        pd.read_csv(
            MODEL_DIR / "feature_order.csv"
        )["feature"]
        .tolist()
    )

    impute_values_df = pd.read_csv(
        MODEL_DIR / "impute_values.csv"
    )

    medians = dict(
        zip(
            impute_values_df["column"],
            impute_values_df["median"]
        )
    )

    hub_segments = pd.read_csv(
        OUTPUT_DIR / "hub_segments.csv"
    )



    clean_shipment, _ = clean_shipments(
        new_raw,
        medians=medians
    )



    features_df = build_features(
        clean_shipment,
        hubs
    )

    X_new = features_df[feature_order]



    probabilities = model.predict_proba(
        X_new
    )[:, 1]

    rounded_probabilities = np.round(
        probabilities,
        4
    )

    flagged = (
        rounded_probabilities >= threshold
    ).astype(int)



    segment_lookup = (
        hub_segments
        .set_index("hub_id")["segment"]
    )

    results = features_df[
        ["shipment_id", "hub_id"]
    ].copy()

    results["segment"] = (
        results["hub_id"].map(segment_lookup)
    )

    results["delay_probability"] = rounded_probabilities
    results["flagged"] = flagged



    escalation_condition = (
        (results["flagged"] == 1) &
        (results["segment"] == "Escalate")
    )

    proactive_call_condition = (
        results["flagged"] == 1
    )

    results["action"] = np.select(
        [
            escalation_condition,
            proactive_call_condition
        ],
        [
            "Escalate to hub manager",
            "Proactive customer call"
        ],
        default="No action"
    )



    results = results[
        [
            "shipment_id",
            "hub_id",
            "segment",
            "delay_probability",
            "flagged",
            "action"
        ]
    ]

    results.to_csv(
        OUTPUT_DIR / "new_shipment_actions.csv",
        index=False,
        float_format="%.4f"
    )


if __name__ == "__main__":
    main()
