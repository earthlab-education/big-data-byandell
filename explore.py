"""
merge_ndvi_cdc: Merge NDVI and CDC data
plot_ndvi_stats: Plot NDVI and CDC data
vartrans: 
"""
def merge_ndvi_cdc(tract_cdc_gdf, ndvi_stats_df):
    """
    Merge NDVI and CDC data.

    Args:
        tract_cdc_gdf (gdf): CDC tracts
        ndvi_stats_df (df): NDVI stats on tracts
    Returns:
        ndvi_cdc_gdf (gdf): merged data as gdf
    """
    ndvi_cdc_gdf = (
        tract_cdc_gdf
        .merge(
            ndvi_stats_df,
            left_on='tract2010', right_on='tract', how='inner')
    )
    return ndvi_cdc_gdf

# ndvi_cdc_gdf = merge_ndvi_cdc(tract_cdc_gdf, ndvi_stats_df)
    
def plot_ndvi_stats(ndvi_cdc_gdf):
    """
    Plot NDVI and CDC data.

    Args:
        ndvi_cdc_gdf (gdf): merged data as gdf
    Returns:
        None
    """
    import geoviews as gv
    from cartopy import crs as ccrs

    # Plot chloropleths with vegetation statistics
    def plot_chloropleth(gdf, **opts):
        """Generate a chloropleth with the given color column"""
        return gv.Polygons(
            gdf.to_crs(ccrs.Mercator()),
            crs=ccrs.Mercator()
        ).opts(xaxis=None, yaxis=None, colorbar=True, **opts)

    (
        plot_chloropleth(
            ndvi_cdc_gdf, color='asthma', cmap='viridis')
        + 
        plot_chloropleth(ndvi_cdc_gdf, color='edge_density', cmap='Greens')
    )
    
# plot_ndvi_stats(tract_cdc_gdf, ndvi_stats_df)
 
def var_trans(ndvi_cdc_gdf):
    """
    Variable Selection and Transformation

    Args:
        ndvi_cdc_gdf (gdf): combined CDC and NDVI gdf
    Returns:
        model_df (df): model DataFrame
    """
    # Variable selection and transformation
    model_df = (
        ndvi_cdc_gdf
        .copy()
        [['frac_veg', 'asthma', 'mean_patch_size', 'edge_density', 'geometry']]
        .dropna()
    )

    model_df['log_asthma'] = np.log(model_df.asthma)
    
    return model_df

# model_df = var_trans(ndvi_cdc_gdf)

def plot_var_trans(model_df):
    """
    Plot matrix to check for transformations

    Args:
        model_df (df): model DataFrame
    Returns:
        ndvi_cdc_hv (hvplot): plot
    """
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray

    # Plot scatter matrix to identify variables that need transformation
    ndvi_cdc_hv = hvplot.scatter_matrix(
        model_df
        [[ 
            'mean_patch_size',
            'edge_density',
            'log_asthma'
        ]]
        )
    return ndvi_cdc_hv
    
# ndvi_cdc_hv = vartrans(ndvi_cdc_gdf)
# ndvi_cdc_hv

def predout(model_df):
    """
    Model fit.

    Args:
        model_df (df): model DataFrame
    Returns:
        y_text (nparray): test dataset
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Select predictor and outcome variables

    X = model_df[['edge_density', 'mean_patch_size']]
    y = model_df[['log_asthma']]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Fit a linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predict asthma values for the test dataset
    y_test['pred_asthma'] = np.exp(reg.predict(X_test))
    y_test['asthma'] = np.exp(y_test.log_asthma)

def plot_pred(y_test):
    """
    Plot test fit.

    Args:
        y_text (nparray): test dataset
    """
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray

    # Plot measured vs. predicted asthma prevalence with a 1-to-1 line
    y_max = y_test.asthma.max()
    (
        y_test
        .hvplot.scatter(
            x='asthma', y='pred_asthma',
            xlabel='Measured Adult Asthma Prevalence', 
            ylabel='Predicted Adult Asthma Prevalence',
            title='Linear Regression Performance - Testing Data'
        )
        .opts(aspect='equal', xlim=(0, y_max), ylim=(0, y_max), height=600, width=600)
    ) * hv.Slope(slope=1, y_intercept=0).opts(color='black')

