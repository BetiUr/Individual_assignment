import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from data_cleaning import clean_data, standardize_columns
from data_insertion import load_data

# Constants
EARTH_RADIUS_KM = 6371.0
EPS_KM = 0.015
MIN_SAMPLES = 1


def cluster_chunk(chunk_df):
    """Run DBSCAN on a chunk of stations and return DataFrame with clusters."""
    if chunk_df.empty:
        chunk_df = chunk_df.copy()
        chunk_df['cluster'] = pd.Series(dtype=int)
        return chunk_df

    coords_rad = chunk_df[['lat', 'lng']].apply(np.radians)
    db = DBSCAN(eps=EPS_KM / EARTH_RADIUS_KM, min_samples=MIN_SAMPLES, metric='haversine')
    labels = db.fit_predict(coords_rad)
    chunk_df = chunk_df.copy()
    chunk_df['cluster'] = labels
    return chunk_df


def split_data_by_rows(stations, n_chunks):
    """Split stations evenly into n_chunks by row count."""
    chunk_size = int(np.ceil(len(stations) / n_chunks))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = stations.iloc[start_idx:end_idx].copy()
        chunks.append(chunk)
    return chunks


def cluster_month_stations(df, num_cpus=8):
    """Cluster stations for a single month using parallel DBSCAN."""
    print("[INFO] Cleaning data...")
    df = standardize_columns(df)
    df = clean_data(df)

    print("[INFO] Extracting station coordinates...")
    with tqdm(total=2, desc="Extracting start/end stations") as pbar:
        start_stations = df[['start_station_name', 'start_lat', 'start_lng']].rename(
            columns={'start_station_name': 'name', 'start_lat': 'lat', 'start_lng': 'lng'}
        )
        pbar.update(1)
        end_stations = df[['end_station_name', 'end_lat', 'end_lng']].rename(
            columns={'end_station_name': 'name', 'end_lat': 'lat', 'end_lng': 'lng'}
        )
        pbar.update(1)

    stations = pd.concat([start_stations, end_stations])
    stations.dropna(subset=['name', 'lat', 'lng'], inplace=True)

    print("[INFO] Removing duplicate stations...")
    stations = stations.drop_duplicates(subset=['name', 'lat', 'lng']).reset_index(drop=True)
    tqdm.write(f"[INFO] Unique station entries: {len(stations)}")

    if stations.empty:
        print("[WARN] No valid stations to cluster.")
        return pd.DataFrame()

    # Split evenly by rows instead of spatial splitting
    chunks = split_data_by_rows(stations, num_cpus)

    clustered_chunks = process_map(
        cluster_chunk,
        chunks,
        max_workers=num_cpus,
        chunksize=1,
        desc="Clustering chunks"
    )

    # Offset clusters to avoid label collisions
    print("[INFO] Offsetting cluster labels...")
    offset = 0
    for i in tqdm(range(len(clustered_chunks)), desc="Offsetting clusters"):
        df_chunk = clustered_chunks[i]
        mask = df_chunk['cluster'] != -1
        if mask.any():
            df_chunk.loc[mask, 'cluster'] += offset
            offset += df_chunk.loc[mask, 'cluster'].max() + 1
        clustered_chunks[i] = df_chunk

    clustered = pd.concat(clustered_chunks, ignore_index=True)

    # Fixing naming conflicts
    print("[INFO] Resolving duplicate station names in different clusters...")
    name_cluster_counts = clustered.groupby('name')['cluster'].nunique()
    split_names = name_cluster_counts[name_cluster_counts > 1].index.tolist()

    if split_names:
        tqdm.write(f"[INFO] Found {len(split_names)} conflicting station names")
        name_to_cluster = {}
        for name in tqdm(split_names, desc="Resolving conflicts"):
            clusters = clustered.loc[clustered['name'] == name, 'cluster']
            name_to_cluster[name] = clusters.value_counts().idxmax()

        mask = clustered['name'].isin(name_to_cluster)
        clustered.loc[mask, 'cluster'] = clustered.loc[mask, 'name'].map(name_to_cluster)

    # Summary
    num_clusters = clustered['cluster'].nunique() - (-1 in clustered['cluster'].unique())
    noise_count = (clustered['cluster'] == -1).sum()
    tqdm.write(f"[CLUSTER] Clusters this month (excluding noise): {num_clusters}")
    tqdm.write(f"[CLUSTER] Noise points: {noise_count}")

    return clustered


if __name__ == "__main__":
    data_dir = r'C:\Users\Beatrice\Downloads\bike_rent_data'
    output_dir = r'C:\Users\Beatrice\Downloads\bike_rent_data\clustered_monthly'
    os.makedirs(output_dir, exist_ok=True)

    monthly_data = load_data(data_dir)

    print("[INFO] Starting monthly clustering...\n")
    for month in tqdm(sorted(monthly_data.keys()), desc="Processing months"):
        print(f"\n[INFO] === Clustering stations for month {month} ===")
        df = monthly_data[month]
        clustered = cluster_month_stations(df, num_cpus=8)

        if clustered.empty:
            print(f"[WARN] No clustered data for month {month}, skipping save.")
            continue

        # Save clustered stations for each month
        out_path = os.path.join(output_dir, f"clustered_stations_{month}.parquet")
        clustered.to_parquet(out_path)
        print(f"[INFO] Saved clustered stations for month {month} at {out_path}")

    print("\n[INFO] All months processed.")
