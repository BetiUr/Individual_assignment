import os
import pandas as pd
import folium
from matplotlib import cm, colors
import matplotlib.pyplot as plt

from data_insertion import load_data
from data_cleaning import clean_data, standardize_columns  # import cleaning funcs

# ---------- CONFIGURATION ----------
DATA_DIR = r'C:\Users\Beatrice\Downloads\bike_rent_data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'station_maps')
CLUSTERED_DIR = os.path.join(DATA_DIR, 'clustered_monthly')
SUMMARY_OUTPUT_PATH = os.path.join(DATA_DIR, 'station_maps', "stations_clusters_summary.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_usage_and_coords(df):
    start_counts = df['start_station_name'].value_counts().reset_index()
    start_counts.columns = ['station_name', 'start_count']

    end_counts = df['end_station_name'].value_counts().reset_index()
    end_counts.columns = ['station_name', 'end_count']

    usage = pd.merge(start_counts, end_counts, on='station_name', how='outer').fillna(0)
    usage['total_usage'] = usage['start_count'] + usage['end_count']

    start_coords = df[['start_station_name', 'start_lat', 'start_lng']].copy()
    start_coords.columns = ['station_name', 'lat', 'lng']

    end_coords = df[['end_station_name', 'end_lat', 'end_lng']].copy()
    end_coords.columns = ['station_name', 'lat', 'lng']

    combined = pd.concat([start_coords, end_coords], ignore_index=True)

    avg_coords = (
        combined.groupby('station_name')[['lat', 'lng']]
        .mean()
        .reset_index()
    )

    merged = pd.merge(usage, avg_coords, on='station_name', how='inner')
    return merged.sort_values('total_usage', ascending=True)

def create_color_map(values, cmap_name='viridis'):
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(cmap_name)
    return [colors.to_hex(cmap(norm(v))) for v in values]

def plot_station_map(df, month, output_dir):
    df['color'] = create_color_map(df['total_usage'])

    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=12)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=5,
            color=row['color'],
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(f"{row['station_name']}<br>Used {row['total_usage']} times", max_width=200),
            tooltip=row['station_name']
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 5px;
        font-size: 12px;
    ">
        <b>Station Popularity</b><br>
        <i style='background: #440154;'>&nbsp;&nbsp;&nbsp;</i> Low<br>
        <i style='background: #21918c;'>&nbsp;&nbsp;&nbsp;</i> Medium<br>
        <i style='background: #fde725;'>&nbsp;&nbsp;&nbsp;</i> High
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path = os.path.join(output_dir, f'2021_{month}_station_popularity_map.html')
    m.save(output_path)
    print(f"[✓] Map saved: {output_path}")

def save_summary_image(summary_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.title("DBSCAN Clusters vs Stations Summary per Month", fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Saved summary image: {output_path}")

def load_cluster_counts(clustered_dir):
    """Load cluster counts for each month from parquet cluster files."""
    cluster_counts = {}
    for file in os.listdir(clustered_dir):
        if not file.endswith('.parquet'):
            continue
        month = file.replace('clustered_stations_', '').replace('.parquet', '')
        path = os.path.join(clustered_dir, file)
        df = pd.read_parquet(path)
        valid = df[df['cluster'] != -1]
        cluster_counts[month] = valid['cluster'].nunique()
    return cluster_counts

if __name__ == "__main__":
    monthly_data = load_data(DATA_DIR)
    cluster_counts = load_cluster_counts(CLUSTERED_DIR)

    summary_list = []

    for month, df in monthly_data.items():
        print(f"\n[INFO] Processing month: {month}")

        df = standardize_columns(df)
        df = clean_data(df)

        summary_df = compute_usage_and_coords(df)
        plot_station_map(summary_df, month, OUTPUT_DIR)

        num_stations = summary_df['station_name'].nunique()
        num_clusters = cluster_counts.get(month, None)
        percent_detected = round(num_clusters / num_stations * 100, 2) if num_clusters is not None and num_stations > 0 else None

        summary_list.append({
            'Month': month,
            'Clusters by DBSCAN': num_clusters if num_clusters is not None else 'N/A',
            'Different Stations': num_stations,
            'Percentage Detected (%)': percent_detected if percent_detected is not None else 'N/A'
        })

    summary_table = pd.DataFrame(summary_list).sort_values('Month')
    print("\nSummary of clusters and stations by month:")
    print(summary_table.to_string(index=False))

    save_summary_image(summary_table, SUMMARY_OUTPUT_PATH)
