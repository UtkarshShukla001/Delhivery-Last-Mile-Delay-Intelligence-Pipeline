import pandas as pd
import numpy as np

from pathlib import Path

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

raw_rows = len(raw)

deduplicated = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(deduplicated)
rows_after_dedup = len(deduplicated)

missing_distance_km_before = (
    deduplicated["distance_km"].isna().sum()
)

missing_parcel_weight_before = (
    deduplicated["parcel_weight_kg"].isna().sum()
)

missing_declared_value_before = (
    deduplicated["declared_value"].isna().sum()
)

missing_payment_mode_before = (
    deduplicated["payment_mode"].isna().sum()
)

sentinel_weight_rows = (
    deduplicated["parcel_weight_kg"] > 100
).sum()

service_type_spellings_before = (
    deduplicated["service_type"].nunique()
)

clean_shipment, medians = clean_shipments(raw)

service_type_spellings_after = (
    clean_shipment["service_type"].nunique()
)

modelling_table = build_features(clean_shipment, hubs)

rows_after_hub_merge = len(modelling_table)

modelling_table["is_delayed"] = (
    modelling_table["delivery_hours"] >
    modelling_table["promised_hours"]
).astype(int)

modelling_table.to_csv(
    DATA_DIR / "modelling_table.csv",
    index=False
)


required_columns = FEATURES + [
    "delivery_hours",
    "is_delayed"
]

nulls_in_features_and_target = (
    modelling_table[required_columns].isna().sum().sum()
)

delayed_count = int(modelling_table["is_delayed"].sum())

delay_rate = modelling_table["is_delayed"].mean()

hours_q1 = modelling_table["delivery_hours"].quantile(0.25)
hours_q3 = modelling_table["delivery_hours"].quantile(0.75)

iqr = hours_q3 - hours_q1

hours_lower_bound = hours_q1 - (1.5 * iqr)
hours_upper_bound = hours_q3 + (1.5 * iqr)

hours_outlier_count = (
    (modelling_table["delivery_hours"] < hours_lower_bound) |
    (modelling_table["delivery_hours"] > hours_upper_bound)
).sum()


declared_value_mean_inr = (
    modelling_table["declared_value_inr"].mean()
)

declared_value_median_inr = (
    modelling_table["declared_value_inr"].median()
)

log_declared_value_mean = (
    modelling_table["log_declared_value"].mean()
)

log_declared_value_median = (
    modelling_table["log_declared_value"].median()
)

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),

    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),

    ("sentinel_weight_rows", sentinel_weight_rows),

    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),

    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),

    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),

    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),

    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),

    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median)
]

checks_df = pd.DataFrame(
    checks,
    columns=["metric", "value"]
)

checks_df.to_csv(
    OUTPUT_DIR / "part_a_checks.csv",
    index=False,
    float_format="%.4f"
)


def create_delay_summary(data, group_column, dimension_name):

    summary = (
        data.groupby(group_column, observed=True)
        .agg(
            shipments=("shipment_id", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean")
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )

    summary.insert(0, "dimension", dimension_name)

    return summary



service_summary = create_delay_summary(
    modelling_table,
    "service_type",
    "service_type"
)


hub_summary = create_delay_summary(
    modelling_table,
    "hub_tier",
    "hub_tier"
)


distance_data = modelling_table.copy()

distance_data["distance_band"] = pd.cut(
    distance_data["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True
)


distance_summary = create_delay_summary(
    distance_data,
    "distance_band",
    "distance_band"
)


delay_by_group = pd.concat(
    [
        service_summary,
        hub_summary,
        distance_summary
    ],
    ignore_index=True
)


delay_by_group.to_csv(
    OUTPUT_DIR / "part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f"
)


numeric_data = modelling_table.select_dtypes(
    include=[np.number]
)

correlation_matrix = numeric_data.corr()

correlation_output = (
    correlation_matrix
    .rename_axis("feature")
    .reset_index()
)

correlation_output.to_csv(
    OUTPUT_DIR / "part_a_correlations.csv",
    index=False,
    float_format="%.4f"
)
