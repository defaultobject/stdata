import pickle
import datetime
import geopandas as gpd

def datetime_to_epoch(datetime):
    """
        Converts a datetime to a number
        args:
            datatime: is a pandas column
    """
    return datetime.astype('int64')//1e9

def timestamp_to_epoch(ts):
    """ Seconds from epoch """
    dt = ts.to_pydatetime()
    return (dt-datetime.datetime(1970,1,1)).total_seconds()

def save_to_pickle(data, name):
    """ Save data to a pickle under the file path name"""
    with open(name, 'wb') as file:
        pickle.dump(data, file)

def read_pickle():
    pass

def to_gdf(df):
    """ Assumes lat/lon cols and espg 4326"""
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(
            df['lon'], 
            df['lat']
        )
    )

    gdf = gdf.set_crs(epsg=4326)

    return gdf
