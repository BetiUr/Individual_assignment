import os
import pandas as pd
import folium
import matplotlib.cm as cm
from matplotlib import colors
from tqdm import tqdm
from branca.element import Template, MacroElement  # for folium legend
import matplotlib.pyplot as plt

# Constants
MAX_POINTS_PER_CLUSTER = 500
MAX_TOTAL_POINTS = 20000
BUBBLE_RADIUS = 200  # Fixed radius for all summary bubbles (in meters)

# Paths
clustered_dir = r'C:\Users\Beatrice\Downloads\bike_rent_data\clustered_monthly'
plots_dir = r'C:\Users\Beatrice\Downloads\bike_rent_data\plots'
os.makedirs(plots_dir, exist_ok=True)


def generate_colors(n):
    cmap = cm.get_cmap("tab20", n)
    return [f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}' for rgba in cmap(range(n))]


def save_overall_summary_image(summary_df, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.title("Number of Clusters per Month", fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Saved overall cluster summary image: {output_path}")


# Global list to store summary data per month
summary_list = []

for file in tqdm(sorted(os.listdir(clustered_dir))):
    if not file.endswith('.parquet'):
        continue

    month = file.replace('clustered_stations_', '').replace('.parquet', '')
    path = os.path.join(clustered_dir, file)
    df = pd.read_parquet(path)

    if df.empty:
        print(f"[WARN] Empty clustered data for {month}, skipping.")
        continue

    valid = df[df['cluster'] != -1]
    num_clusters = valid['cluster'].nunique()
    print(f"\nMonth {month}: {len(df)} points | {num_clusters} valid clusters")

    # Save monthly cluster count for overall summary
    summary_list.append({'Month': month, 'Clusters': num_clusters})

    cluster_ids = sorted(valid['cluster'].unique())
    colors_list = generate_colors(num_clusters)
    color_map = dict(zip(cluster_ids, colors_list))

    # --- Map 1: Detailed points with popups ---
    m1 = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=12)
    total_points = 0

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]

        if len(cluster_data) > MAX_POINTS_PER_CLUSTER:
            cluster_data = cluster_data.sample(n=MAX_POINTS_PER_CLUSTER, random_state=42)

        if total_points + len(cluster_data) > MAX_TOTAL_POINTS:
            break

        total_points += len(cluster_data)
        color = color_map.get(cluster_id, 'gray')

        if cluster_id != -1:
            unique_station_names = sorted(cluster_data['name'].unique())
            centroid = cluster_data[['lat', 'lng']].mean()

            popup_html = "<b>Cluster {} Stations:</b><br>{}".format(
                cluster_id, "<br>".join(unique_station_names)
            )

            folium.Marker(
                location=[centroid['lat'], centroid['lng']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m1)

        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.5
            ).add_to(m1)

    map1_path = os.path.join(plots_dir, f"{month}_detailed_clusters.html")
    m1.save(map1_path)
    print(f"Saved detailed map: {map1_path}")

    # --- Map 2: Summary circles for each cluster, color by popularity (number of points), fixed bubble size ---
    m2 = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=12)

    popularity_points = valid.groupby('cluster').size()
    pop_min, pop_max = popularity_points.min(), popularity_points.max()

    norm = colors.Normalize(vmin=pop_min, vmax=pop_max)
    cmap = cm.get_cmap('viridis')
    pop_color_map = {cid: colors.to_hex(cmap(norm(popularity_points[cid]))) for cid in popularity_points.index}

    for cluster_id in cluster_ids:
        cluster_data = valid[valid['cluster'] == cluster_id]
        centroid = cluster_data[['lat', 'lng']].mean()
        num_points = popularity_points[cluster_id]
        color = pop_color_map[cluster_id]
        label = f"Cluster {cluster_id} ({num_points} points)<br>Stations:<br>" + "<br>".join(sorted(cluster_data['name'].unique()))

        folium.Circle(
            location=[centroid['lat'], centroid['lng']],
            radius=BUBBLE_RADIUS,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(label, max_width=300)
        ).add_to(m2)

    # Add color legend to map 2 (summary map)
    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 180px; height: 100px; 
        background: white; 
        border:2px solid grey; 
        z-index:9999; 
        font-size:14px;
        padding: 10px;
        box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        ">
        <b>Cluster Popularity</b><br>
        <i>Points per cluster</i><br>
        <div style="background: linear-gradient(to right, 
            {colors.to_hex(cmap(0))}, 
            {colors.to_hex(cmap(0.25))}, 
            {colors.to_hex(cmap(0.5))}, 
            {colors.to_hex(cmap(0.75))}, 
            {colors.to_hex(cmap(1.0))}
            ); 
            height: 20px; width: 100%; margin: 5px 0;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span>{pop_min}</span><span>{pop_max}</span>
        </div>
    </div>
    '''

    legend = MacroElement()
    legend._template = Template(legend_html)
    m2.get_root().add_child(legend)

    map2_path = os.path.join(plots_dir, f"{month}_summary_clusters.html")
    m2.save(map2_path)
    print(f"Saved summary map: {map2_path}")


# After processing all months, create and save overall summary table image
summary_df = pd.DataFrame(summary_list).sort_values('Month')
print("\nOverall Cluster Counts per Month:")
print(summary_df.to_string(index=False))

summary_image_path = os.path.join(plots_dir, "overall_clusters_summary.png")
save_overall_summary_image(summary_df, summary_image_path)
