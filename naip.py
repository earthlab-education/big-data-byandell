"""
naip_path: Create NAIP tracts path
download_naip_tracts: Download NAIP Tracts
ndvi_naip_one: Get stats for one NAIP tract
ndvi_naip_df: Get stats for all NAIP tracts
"""
def naip_path(data_dir, place = 'chicago'):
    """
    Create NAIP tracts path.
    
    Args:
        data_dir (str): data directory
        place (str): name of place
    Returns:
        naip_stats_path (str): address of NAIP tracts
    """
    import os
    
    naip_stats_path = os.path.join(data_dir, f'{place}-naip-stats.csv')

    return naip_stats_path

# naip_stats_path = naip_path(data_dir, 'chicago')

def download_naip_tracts(naip_stats_path, tract_cdc_gdf):
    """
    Download NAIP Tracts.
    
    Args:
        naip_path (str): NAIP tracts directory
        tract_cdc_gdf (gdf): gdf of combined place and disease
    Returns:
        naip_stats_tracts (ndarray): naip tract values
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

    return naip_stats_df

# naip_stats_path = naip_path(data_dir, 'chicago')
# naip_stats_df = download_naip_tracts(naip_stats_path, tract_cdc_gdf)
# naip_stats_df.tract.values

def ndvi_naip_one(tract_cdc_gdf, tract, tract_date_gdf):
    """
    Get stats for one NAIP tract.

    Args:
        tract_cdc_gdf (gdf): gdf
        tract (int): tract number
        tract_date_gdf (gdf): gdf

    Returns:
        _type_: _description_
    """
    import numpy as np
    import rioxarray as rxr
    import rioxarray.merge as rxrmerge
    from scipy.ndimage import label
    from scipy.ndimage import convolve

    # Open all images for tract
    tile_das = []
    for _, href_s in tract_date_gdf.iterrows():
        # Open vsi connection to data
        tile_da = rxr.open_rasterio(
            href_s.rgbir_href, masked=True).squeeze()
        
        # Clip data
        boundary = (
            tract_cdc_gdf
            .set_index('tract2010')
            .loc[[tract]]
            .to_crs(tile_da.rio.crs)
            .geometry
        )
        crop_da = tile_da.rio.clip_box(
            *boundary.envelope.total_bounds,
            auto_expand=True)
        clip_da = crop_da.rio.clip(boundary, all_touched=True)
            
        # Compute NDVI
        ndvi_da = (
            (clip_da.sel(band=4) - clip_da.sel(band=1)) 
            / (clip_da.sel(band=4) + clip_da.sel(band=1))
        )

        # Accumulate result
        tile_das.append(ndvi_da)

    # Merge data
    scene_da = rxrmerge.merge_arrays(tile_das)

    # Mask vegetation
    veg_mask = (scene_da>.3)

    # Calculate statistics and save data to file
    total_pixels = scene_da.notnull().sum()
    veg_pixels = veg_mask.sum()

    # Calculate mean patch size
    labeled_patches, num_patches = label(veg_mask)
    # Count patch pixels, ignoring background at patch 0
    patch_sizes = np.bincount(labeled_patches.ravel())[1:] 
    mean_patch_size = patch_sizes.mean()

    # Calculate edge density
    kernel = np.array([
        [1, 1, 1], 
        [1, -8, 1], 
        [1, 1, 1]])
    edges = convolve(veg_mask, kernel, mode='constant')
    edge_density = np.sum(edges != 0) / veg_mask.size
    
    return total_pixels, veg_pixels, mean_patch_size, edge_density

# total_pixels, veg_pixels, mean_patch_size, edge_density = ndvi_naip_one(tract_cdc_gdf, tract, tract_date_gdf)
    
def ndvi_naip_df(naip_stats_path, tract_cdc_gdf, scene_df):
    """
    Get stats for all NAIP tracts.

    Args:
        naip_stats_path (str): address of NAIP tracts
        tract_cdc_gdf (gdf): gdf of CDC tracts
        scene_df (df): df of scene
    Returns:
        ndvi_stats_df (df): NDVI stats DataFrame
    """
    import os
    import pandas as pd
    from tqdm.notebook import tqdm
    
    # Skip this step if data are already downloaded 
    if not scene_df is None:
        # Loop through the census tracts with URLs
        for tract, tract_date_gdf in tqdm(scene_df.groupby('tract')):
            total_pixels, veg_pixels, mean_patch_size, edge_density = ndvi_naip_one(tract_cdc_gdf, tract, tract_date_gdf)
            # Add a row to the statistics file for this tract
            pd.DataFrame(dict(
                tract=[tract],
                total_pixels=[int(total_pixels)],
                frac_veg=[float(veg_pixels/total_pixels)],
                mean_patch_size=[mean_patch_size],
                edge_density=[edge_density]
            )).to_csv(
                naip_stats_path, 
                mode='a', 
                index=False, 
                header=(not os.path.exists(naip_stats_path))
            )

    # Re-load results from file
    ndvi_stats_df = pd.read_csv(naip_stats_path)
    
    return ndvi_stats_df

# ndvi_stats_df = ndvi_naip_df(naip_stats_path, tract_cdc_gdf, scene_df)
