"""
merge_ndvi_cdc: Merge NDVI and CDC data
plot_ndvi_index: Plot NDVI and CDC data
vartrans: Variable Selection and Transformation
hvplot_matrix: HV plot of model matrixtrain_test: Model fit using train and test sets
plot_train_test: Plot test fit
"""
def merge_ndvi_cdc(tract_cdc_gdf, ndvi_index_df):
    """
    Merge NDVI and CDC data.

    Args:
        tract_cdc_gdf (gdf): CDC tracts
        ndvi_index_df (df): NDVI stats on tracts
    Returns:
        ndvi_cdc_gdf (gdf): merged data as gdf
    """
    ndvi_cdc_gdf = (
        tract_cdc_gdf
        .merge(
            ndvi_index_df,
            left_on='tract2010', right_on='tract', how='inner')
    )
    return ndvi_cdc_gdf

# ndvi_cdc_gdf = merge_ndvi_cdc(tract_cdc_gdf, ndvi_index_df)

def plot_chloropleth(gdf, **opts):
    """
    Generate a chloropleth with the given color column.
    
    Args:
        gdf (gdf): GeoDataFrame
    Returns:
        _ (gv_plot): plot
    """
    import geoviews as gv
    from cartopy import crs as ccrs
    
    return gv.Polygons(
        gdf.to_crs(ccrs.Mercator()),
        crs=ccrs.Mercator()
    ).opts(xaxis=None, yaxis=None, colorbar=True, **opts)
    
# plot_chloropleth(gdf)
    
def plot_ndvi_index(ndvi_cdc_gdf):
    """
    Plot NDVI and CDC data.

    Args:
        ndvi_cdc_gdf (gdf): merged data as gdf
    Returns:
        None
    """
    plot_ndvi = (
        plot_chloropleth(ndvi_cdc_gdf, color='asthma', cmap='viridis', title='Asthma')
        + 
        plot_chloropleth(ndvi_cdc_gdf, color='edge_density', cmap='Greens', title='Edge Density')
    )
    
    return plot_ndvi
    
# plot_ndvi_index(tract_cdc_gdf, ndvi_index_df)
 
def var_trans(ndvi_cdc_gdf):
    """
    Variable Selection and Transformation

    Args:
        ndvi_cdc_gdf (gdf): combined CDC and NDVI gdf
    Returns:
        model_df (df): model DataFrame
    """
    import numpy as np
    
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

def hvplot_matrix(model_df):
    """
    HV plot of model matrix

    Args:
        model_df (df): model DataFrame
    Returns:
        matrix_hv (hvplot): plot
    """
    import holoviews as hv
    import hvplot.pandas
    import hvplot.xarray

    # Plot scatter matrix to identify variables that need transformation
    matrix_hv = hvplot.scatter_matrix(
        model_df
        [[ 
            'mean_patch_size',
            'edge_density',
            'log_asthma'
        ]]
        )
    
    return matrix_hv
    
# hvplot_matrix(ndvi_cdc_gdf)

def train_test(model_df, test_size=0.33, random_state=42):
    """
    Model fit using train and test sets.

    Args:
        model_df (df): model DataFrame
    Returns:
        y_text (nparray): test dataset
        reg (LinearRegression): LinearRegression object
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Select predictor and outcome variables

    X = model_df[['edge_density', 'mean_patch_size']]
    y = model_df[['log_asthma']]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Fit a linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Predict asthma values for the test dataset
    y_test['pred_asthma'] = np.exp(reg.predict(X_test))
    y_test['asthma'] = np.exp(y_test.log_asthma)
    
    return y_test, reg

# y_test, reg = trait_test(model_df)

def plot_train_test(y_test):
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
    
    hv_test = (
        y_test
        .hvplot.scatter(
            x='asthma', y='pred_asthma',
            xlabel='Measured Adult Asthma Prevalence', 
            ylabel='Predicted Adult Asthma Prevalence',
            title='Linear Regression Performance - Testing Data'
        )
        .opts(aspect='equal', xlim=(0, y_max), ylim=(0, y_max), height=600, width=600)
    ) * hv.Slope(slope=1, y_intercept=0).opts(color='black')

    return hv_test

# plot_train_test(y_test)

def plot_resid(model_df, reg, yvar='log_asthma', xvar=['edge_density', 'mean_patch_size']):
    """
    Plot model residual
    
    Args:
        model_df (df): model object
        reg (LinearRegression): LinearRegression object
        yvar (str, optional): y variable name. Defaults to 'asthma'.
    Returns:
        resid_gv (gv_plot): plot
    """
    import numpy as np
    
    model_df[f'pred_{yvar}'] = np.exp(reg.predict(model_df[xvar]))
    model_df['err_yvar'] = model_df[f'pred_{yvar}'] - model_df[yvar]

    # Plot error geographically as a chloropleth
    resid_gv = (
        plot_chloropleth(model_df, color='err_yvar', cmap='RdBu', title="Residuals for Asthma")
        .redim.range(err_yvar=(-.3, .3))
        #.opts(frame_width=600, aspect='equal')
    )
    
    return resid_gv

# plot_resid(model_df, yvar='asthma')
