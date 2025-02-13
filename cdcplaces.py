"""
shp_tract_path: Set tract path
download_census_tract: Download the census tracts
hvplot_tract_gdf: HV plot census tracts with satellite imagery background
download_cdc_disease: Download CDC Disease data
join_tract_cdc: Join Census Tract and CDC Disease Data
plot_gdf_gv: Plot asthma data as chloropleth
"""
def shp_tract_path(data_dir, place = 'chicago-tract'):
    """
    Set tract path.
    
    Args:
        data_dir (str): data directory
        place (str): name for place directory
    
    Returns:
        tract_path (str): shapefile address
    """
    import os
    tract_dir = os.path.join(data_dir, place)
    os.makedirs(tract_dir, exist_ok=True)
    tract_path = os.path.join(tract_dir, f'{place}.shp')

    return tract_path

# tract_path = shp_tract_path(place = 'chicago-tract')
    
def download_census_tract(tract_path,
                          placename = 'Chicago'):
    """
    Download the census tracts (only once)
    
    Args:
        tract_path (str): address of tract
        
    Returns:
        place_tract_gdf (GeoDataFrame): gdf for place
    """
    import os
    import geopandas as gpd
    if not os.path.exists(tract_path):
        tract_url = ('https://data.cdc.gov/download/x7zy-2xmx/application%2Fzip')
        tract_gdf = gpd.read_file(tract_url)
        place_tract_gdf = tract_gdf[tract_gdf.PlaceName == placename]
        place_tract_gdf.to_file(tract_path, index=False)

    # Load in the census tract data
    place_tract_gdf = gpd.read_file(tract_path)
    
    return place_tract_gdf
    
# place_tract_gdf = download_census_tract('chicago-tract', 'Chicago')

def hvplot_tract_gdf(place_tract_gdf):
    """
    HV plot census tracts with satellite imagery background.
    
    Args:
        place_tract_gdf (GeoDataFrame): gdf for place
        
    Returns:
        place_hv (hvplot): plot
    """
    import geopandas as gpd
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    from cartopy import crs as ccrs
    
    place_hv = (
        place_tract_gdf
        .to_crs(ccrs.Mercator())
        .hvplot(
            line_color='orange', fill_color=None, 
            crs=ccrs.Mercator(), tiles='EsriImagery',
            frame_width=600)
    )
    return place_hv
    
# hvplot_tract_gdf(place_tract_gdf)

def download_cdc_disease(data_dir,
                         disease = 'asthma',
                         year = '2022',
                         state = 'IL',
                         county = 'Cook',
                         measureid = 'CASTHMA',
                         limit = 1500):
    """
    Download CDC Disease data

    Args:
        data_dir (str): data directory
        disease (str, optional): name of disease
        year (str, optional): data year
        state (str, optional): state abbreviation
        county (str, optional): county name
        measureid (str, optional): ID for disease measure
        limit (str, optional): limit to narrow search

    Returns:
        cdc_df (df): DataFrame
    """
    import os
    import pandas as pd
    
    # Set up a path for the asthma data
    cdc_path = os.path.join(data_dir, f'{disease}.csv')

    # Download asthma data (only once)
    if not os.path.exists(cdc_path):
        cdc_url = (
            "https://data.cdc.gov/resource/cwsq-ngmh.csv"
            f"?year={year}"
            f"&stateabbr={state}"
            f"&countyname={county}"
            f"&measureid={measureid}"
            f"&$limit={limit}"
        )
        cdc_df = (
            pd.read_csv(cdc_url)
            .rename(columns={
                'data_value': 'asthma',
                'low_confidence_limit': 'asthma_ci_low',
                'high_confidence_limit': 'asthma_ci_high',
                'locationname': 'tract'})
            [[
                'year', 'tract', 
                'asthma', 'asthma_ci_low', 'asthma_ci_high', 'data_value_unit',
                'totalpopulation', 'totalpop18plus'
            ]]
        )
        cdc_df.to_csv(cdc_path, index=False)

    # Load in asthma data
    cdc_df = pd.read_csv(cdc_path)

    return cdc_df

# cdc_df = download_cdc_disease(data_dir, disease = 'asthma')

def join_tract_cdc(place_tract_gdf, cdc_df):
    """
    Join Census Tract and CDC Disease Data.

    Args:
        place_tract_gdf (gdf): place image
        cdc_df (df): disease data frame
    
    Returns:
        tract_cdc_gdf (gdf): combined gdf
    """
    # Change tract identifier datatype for merging
    place_tract_gdf.tract2010 = place_tract_gdf.tract2010.astype('int64')

    # Merge census data with geometry
    tract_cdc_gdf = (
        place_tract_gdf
        .merge(cdc_df, left_on='tract2010', right_on='tract', how='inner')
    )
    return tract_cdc_gdf

# tract_cdc_gdf = join_tract_cdc(place_tract_gdf, cdc_df)

def plot_gdf_gv(tract_cdc_gdf):
    """
    Plot asthma data as chloropleth.

    Args:
       tract_cdc_gdf (gdf): combined gdf 
    """
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray
    import geoviews as gv
    from cartopy import crs as ccrs

    tract_cdc_gv = (
        gv.tile_sources.EsriImagery
        * 
        gv.Polygons(
            tract_cdc_gdf.to_crs(ccrs.Mercator()),
            vdims=['asthma', 'tract2010'],
            crs=ccrs.Mercator()
        ).opts(color='asthma', colorbar=True, tools=['hover'])
    ).opts(width=600, height=600, xaxis=None, yaxis=None)

    return tract_cdc_gv

# tract_cdc_gv = plot_gv(tract_cdc_gdf)
