"""
header_csv: Header of CSV file
get_last_row_csv: Check Last Row of CSV File
check_element_in_csv: Check value of element in CSV file
check_naip_tracts: Check if NAIP tracts stored
"""
def header_csv(file_path):
    """
    Header of CSV file.

    Args:
        file_path (str): file path
    Returns:
        header (str): column headers
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    header = df.columns.tolist()
    return header

# header = header_csv(file_path)

def get_last_row_csv(file_path):
    """
    Check Last Row of CSV File.
    
    Args:
        file_path (str): file path
    Returns:
        last_row (str): last row
    """
    import csv
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            # Handle empty CSV
            try:
              *_, last_row = reader
              return last_row
            except ValueError:
              return None
    except FileNotFoundError:
        return "File not found."

# last_row = get_last_row_csv(file_path)

def check_element_in_csv(filename, column_name, target_value):
    """
    Check value of element in CSV file.

    Args:
        filename (str): The path to the CSV file.
        column_name (str): The name of the column to search in.
        target_value: The value to search for.

    Returns:
        bool: True if the element is found, False otherwise.
    """
    import csv

    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[column_name] == str(target_value):
                    return True
        return False
    except FileNotFoundError:
         return False
    

# filename = 'my_data.csv'
# column_name = 'age'
# target_value = 30
# is_found = check_element_in_csv(filename, column_name, target_value):

def check_naip_tracts(naip_index_path, naip_scenes_df):
    """
    Check if NAIP tracts stored.

    Args:
        naip_index_path (str): address of NAIP tracts
        naip_scenes_df (df): df of scenes
    Returns:
        ndvi_stats_df (df): NDVI stats DataFrame
    """
    from tqdm.notebook import tqdm
    
    if not naip_scenes_df is None:
        # Loop through the census tracts with URLs
        for tract, tract_date_gdf in tqdm(naip_scenes_df.groupby('tract')):
            print(tract, check_element_in_csv(naip_index_path, 'tract', tract))
