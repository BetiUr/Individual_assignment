# 🚲 Clustering Evaluation for Detection of Bike Rental Stations and their Popularity 
The point of this project was to apply DBSCAN clustering for detection of bike rental stations and to evaluate if custering can correctly capture majority of them and their popularity.
The evaluation is done by comparing results of clustering with simple grouping of the data by unique station names across 12 different months of year 2021 in Chicago, USA.

## 📌 Features

- **Data Insertion**
    - Code inserts data by month from a chosen directory
    - Data can be found on
https://www.dropbox.com/scl/fo/t8f0rqerchmsvxx2fqkqg/AG_wMKLa1c8M5aAhS7F0xFY/bike_rent.zip?rlkey=ntyfzjyulz2vbz4udbpp81ix8&e=1&dl=0

- **Data Cleaning**
  - Standardizes column names by stripping whitespace and converting them to lowercase.
  - Removes rows with missing values in key columns like station names and coordinates.

- **Monthly Clustering with DBSCAN**
  - Splits data into chunks and runs DBSCAN clustering in parallel using haversine distance.
  - Adjusts cluster labels to avoid overlaps and resolves duplicate station name conflicts.
  - Saves clustered station data monthly as parquet files.
  - Prints progress and summary info throughout.

- **Monthly & Overall Summaries**
  - Loads monthly clustered bike station data.
  - For each month, creates two folium maps:
    • A detailed map showing cluster points with popups.
    • A summary map with colored bubbles sized by cluster popularity and a legend.
  - Saves these maps as HTML files.
  - Generates and saves an overall summary image of cluster counts per month.

  - **Actual station finding by grouping**
    - Loads monthly bike rental data and associated cluster results.
    - Cleans and processes each month’s data to calculate station usage and average coordinates.
    - Creates and saves an interactive folium map showing station popularity with color-coded markers and a legend.
    - Compiles a summary table comparing DBSCAN cluster counts to the number of unique stations per month, including percentage detected.
    - Saves this summary as a PNG image.

---

## 📁 Project Structure
  - data_insertion.py
  - data_cleaning.py
    -  stations_clustering_DBSCAN.py
      - stations_cluster_visualization.py
    - finding_stations_by_grouping.py
   
# 🚀 Clone the repository
git clone https://github.com/BetiUr/Individual_assignment.git
cd Individual_assignment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 📦 Install required libraries
pip install -r requirements.txt


