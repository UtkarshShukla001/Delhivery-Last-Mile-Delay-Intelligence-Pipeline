
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances



ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def main():

    modelling_table = pd.read_csv(
        DATA_DIR / "modelling_table.csv"
    )

    hubs = pd.read_csv(
        DATA_DIR / "hubs.csv"
    )



    hub_profile = (
        modelling_table
        .groupby("hub_id", as_index=False)
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean")
        )
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    hub_profile.to_csv(
        OUTPUT_DIR / "hub_profile.csv",
        index=False,
        float_format="%.4f"
    )


    clustering_profile = pd.read_csv(
        OUTPUT_DIR / "hub_profile.csv"
    )

    profile_columns = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]

    clustering_profile = (
        clustering_profile
        .sort_values("hub_id")
        .reset_index(drop=True)
    )



    scaler = StandardScaler()

    scaled_profiles = scaler.fit_transform(
        clustering_profile[profile_columns]
    )



    separation_rows = []
    cluster_labels_k3 = None

    for k in [2, 3, 4, 5]:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(scaled_profiles)

        distance_matrix = pairwise_distances(
            scaled_profiles,
            metric="euclidean"
        )

        row_indices, col_indices = np.triu_indices(
            len(clustering_profile),
            k=1
        )

        pair_distances = distance_matrix[
            row_indices,
            col_indices
        ]

        same_cluster = (
            labels[row_indices] == labels[col_indices]
        )

        different_cluster = ~same_cluster

        within_mean = pair_distances[same_cluster].mean()
        across_mean = pair_distances[different_cluster].mean()

        across_over_within = across_mean / within_mean

        cluster_sizes = sorted(
            np.bincount(labels).tolist(),
            reverse=True
        )

        separation_rows.append({
            "K": k,
            "across_over_within": across_over_within,
            "cluster_sizes": "-".join(
                str(size) for size in cluster_sizes
            )
        })

        if k == 3:
            cluster_labels_k3 = labels.copy()


    separation_df = pd.DataFrame(
        separation_rows,
        columns=[
            "K",
            "across_over_within",
            "cluster_sizes"
        ]
    )

    separation_df.to_csv(
        OUTPUT_DIR / "part_d_separation.csv",
        index=False,
        float_format="%.4f"
    )



    clustering_profile["cluster"] = cluster_labels_k3

    cluster_delay_rates = (
        clustering_profile
        .groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    ordered_clusters = cluster_delay_rates.index.tolist()

    segment_names = [
        "Standard",
        "Watch",
        "Escalate"
    ]

    cluster_to_segment = dict(
        zip(ordered_clusters, segment_names)
    )

    clustering_profile["segment"] = (
        clustering_profile["cluster"]
        .map(cluster_to_segment)
    )



    hub_segments = clustering_profile[
        ["hub_id", "segment"]
    ].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )

    hub_segments = (
        hub_segments
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    hub_segments = hub_segments[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ]

    hub_segments.to_csv(
        OUTPUT_DIR / "hub_segments.csv",
        index=False
    )



    segment_profile = (
        clustering_profile
        .groupby("segment", as_index=False)
        .agg(
            hubs=("hub_id", "nunique"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean")
        )
    )

    segment_order = {
        "Standard": 0,
        "Watch": 1,
        "Escalate": 2
    }

    segment_profile["_order"] = (
        segment_profile["segment"].map(segment_order)
    )

    segment_profile = (
        segment_profile
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    segment_profile.to_csv(
        OUTPUT_DIR / "segment_profile.csv",
        index=False,
        float_format="%.4f"
    )



    expected_segment_by_tier = {
        1: "Standard",
        2: "Watch",
        3: "Escalate"
    }

    hub_segments["expected_segment"] = (
        hub_segments["hub_tier"]
        .map(expected_segment_by_tier)
    )

    mismatches = hub_segments[
        hub_segments["segment"] !=
        hub_segments["expected_segment"]
    ].copy()

    mismatches = mismatches[
        ["hub_id", "segment", "hub_city", "hub_tier"]
    ]

    mismatches.to_csv(
        OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
        index=False
    )


if __name__ == "__main__":
    main()
