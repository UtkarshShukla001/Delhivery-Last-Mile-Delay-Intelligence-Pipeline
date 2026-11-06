import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    clean_shipment = raw.copy()
    clean_shipment = clean_shipment.drop_duplicates()

    clean_shipment["service_type"] = clean_shipment["service_type"].str.strip().str.title()

    clean_shipment["payment_mode"] = clean_shipment["payment_mode"].fillna("Unknown")

    clean_shipment["declared_value_inr"] = clean_shipment["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    clean_shipment["declared_value_inr"] = pd.to_numeric(clean_shipment["declared_value_inr"], errors="coerce")
    clean_shipment = clean_shipment.drop(columns=["declared_value"])

    clean_shipment["dispatch_time"] = pd.to_datetime(clean_shipment["dispatch_time"], format="%Y-%m-%d %H:%M", errors="coerce")

    clean_shipment.loc[clean_shipment["parcel_weight_kg"]>100, "parcel_weight_kg"] = np.nan

    columns_to_compute_median = ["distance_km", "parcel_weight_kg", "declared_value_inr"]

    if medians is None:
        medians = {
            "distance_km": float(clean_shipment["distance_km"].median()),
            "parcel_weight_kg": float(clean_shipment["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean_shipment["declared_value_inr"].median())
        }
        

    clean_shipment[columns_to_compute_median] = clean_shipment[columns_to_compute_median].fillna(medians)
    return clean_shipment, medians


def build_features(clean_shipment, hubs):
    features_df = clean_shipment.merge(hubs, on="hub_id", how="left", validate="many_to_one")

    features_df["promised_hours"] = features_df["service_type"].map(PROMISED_HOURS)
    features_df["log_declared_value"] = np.log(features_df["declared_value_inr"])
    features_df["is_cod"] = (features_df["payment_mode"] == "COD").astype(int)
    features_df["is_weekend"] = (features_df["dispatch_time"].dt.day_of_week >= 5).astype(int)
    features_df["dispatch_hour"] = features_df["dispatch_time"].dt.hour

    return features_df