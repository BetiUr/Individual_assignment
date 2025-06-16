from data_insertion import load_data  # Assumes this loads monthly data into a dict

def clean_data(df):
    """
    Cleans the input DataFrame by removing rows with missing values in critical columns.
    """
    required_columns = [
        'start_station_name', 'end_station_name',
        'start_lat', 'start_lng', 'end_lat', 'end_lng'
    ]
    before = df.shape[0]
    df_cleaned = df.dropna(subset=required_columns).copy()
    after = df_cleaned.shape[0]
    print(f"[CLEAN] Rows before: {before}, after: {after}")
    return df_cleaned


def standardize_columns(df):
    """
    Standardizes column names by stripping whitespace and converting to lowercase.
    """
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_ride_and_station_info(df):
    """
    Returns the number of rides and a set of all unique stations in the DataFrame.
    """
    ride_count = len(df)
    start_stations = df['start_station_name'].unique()
    end_stations = df['end_station_name'].unique()
    all_stations = set(start_stations) | set(end_stations)
    return ride_count, all_stations
