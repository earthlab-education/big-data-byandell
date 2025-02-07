def download_naip_tracks(data_dir, place="chicago"):
    """
    Download NAIP Tracks
    
    Args:
        data_dir (char): data directory
        place (char): name of place
        
    Returns:
        naip_stats_tracts (df): naip tract values
    """
    import os
    import pandas as pd
    import shapely
    import pystac_client
    from tqdm.notebook import tqdm
    
    # Connect to the planetary computer catalog
    e84_catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    # Convert geometry to lat/lon for STAC
    tract_latlon_gdf = tract_cdc_gdf.to_crs(4326)

    # Define a path to save naip stats
    naip_stats_path = os.path.join(data_dir, f'{place}-naip-stats.csv')

    # Check for existing data - do not access duplicate tracts
    downloaded_tracts = []
    if os.path.exists(naip_stats_path):
        naip_stats_df = pd.read_csv(naip_stats_path)
    else:
        # Download asthma data (only once)
        print('No census tracts downloaded so far')
        # Loop through each census tract
        scene_dfs = []
        for i, tract_values in tqdm(tract_latlon_gdf.iterrows()):
            tract = tract_values.tract2010
            # Check if statistics are already downloaded for this tract
            if not (tract in downloaded_tracts):
                # Retry up to 5 times in case of a momentary disruption
                i = 0
                retry_limit = 5
                while i < retry_limit:
                    # Try accessing the STAC
                    try:
                        # Search for tiles
                        naip_search = e84_catalog.search(
                            collections=["naip"],
                            intersects=shapely.to_geojson(tract_values.geometry),
                            datetime="2021"
                        )
                        
                        # Build dataframe with tracts and tile urls
                        scene_dfs.append(pd.DataFrame(dict(
                            tract=tract,
                            date=[pd.to_datetime(scene.datetime).date() 
                                for scene in naip_search.items()],
                            rgbir_href=[scene.assets['image'].href for scene in naip_search.items()],
                        )))
                        
                        break
                    # Try again in case of an APIError
                    except pystac_client.exceptions.APIError:
                        print(
                            f'Could not connect with STAC server. '
                            f'Retrying tract {tract}...')
                        time.sleep(2)
                        i += 1
                        continue
            
        # Concatenate the url dataframes
        if scene_dfs:
            naip_stats_df = pd.concat(scene_dfs).reset_index(drop=True)
            naip_stats_df.to_csv(naip_stats_path, index=False)
        else:
            naip_stats_df = None

    naip_stats_tracts = naip_stats_df.tract.values

    return naip_stats_tracts

# naip_stats_tracts = download_naip_tracts(data_dir, place = 'chicago')
