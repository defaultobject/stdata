import pickle
import datetime
import geopandas as gpd
from shapely import wkt
import itertools

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

def to_gdf_from_wkt(df, geom='geometry'):
    """ Convert geometry in WKT text to a geometry"""
    gdf = gpd.GeoDataFrame(
        df,
        geometry = gpd.GeoSeries.from_wkt(df[geom])
    )
    return gdf


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

def flatten_pd_index(df, sep='_', copy=True):
    """ Flattens a multi level column in a single one whilst conacatenating the columns """

    if copy:
        df = df.copy()

    df.columns = [sep.join(col).strip(sep) for col in df.columns.values]
    return df

def flatten_2d_list(a):
    try:    
        if type(a) == list:
            return list(itertools.chain(*a))
    except TypeError as e:
        print(e)
        print(a)
        breakpoint()

    return a


