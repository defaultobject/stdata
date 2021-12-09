import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import geopandas as gpd
from geopandas import GeoSeries

from .feature_extraction import _get_unique_spatial_points

def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to):
    return (x - np.mean(wrt_to, axis=0))/np.std(wrt_to, axis=0)

def un_normalise_df(x, wrt_to):
    return x* np.std(wrt_to, axis=0) + np.mean(wrt_to, axis=0)

def train_test_split_indices(N, split=0.5, seed=0):
    """ Compute a random split based on seed """

    np.random.seed(seed)
    rand_index = np.random.permutation(N)

    N_tr =  int(N * split) 

    return rand_index[:N_tr], rand_index[N_tr:] 




def spatial_k_fold(df, num_folds, lat_col='lat', lon_col='lon', group_col='group'):
    """ Split the points of gdf into equal area blocks """
    df = df.copy()

    #assign gid so we can avoid spatial merge later
    df['gid'] = df.index

    sp_gdf = _get_unique_spatial_points(df, lat_col, lon_col)

    # Compute equal sectors around sp_gdf

    #TODO: work directly in radians

    total_angle = 0
    total_angle_radians = 0

    angle = 360/num_folds
    angle_radians = angle * np.pi / 180
    steps = 200

    # Compute center point and radius around data
    x1, y1, x2, y2 = sp_gdf.total_bounds
    center_x = (x1+x2)/2
    center_y = (y1+y2)/2

    x_radius = (x2-x1)/2
    y_radius = (y2-y1)/2

    # Compute sectors with the (approximate) bounding circle
    points = []
    for i in range(num_folds):
        angle = 360/num_folds
        
        sector = []
        for step in range(steps):
            # location on unit circle
            _x = np.cos(total_angle_radians + angle_radians*step/steps)
            _y = np.sin(total_angle_radians + angle_radians*step/steps)
        
            # scale by box widths
            sector.append(
                [center_x + _x * x_radius, center_y + _y * y_radius]
            )
            
        points.append(sector)
        
        total_angle += angle
        total_angle_radians = total_angle * np.pi / 180
        
    # Construct polygons
    # repeat first point so we can loop through without special case
    sectors = []
    for i in range(len(points)):
        poly = Polygon([
            [center_x, center_y],
            *points[i]
        ])
        sectors.append(poly)
        
    sectors = gpd.GeoSeries(sectors)
    
    # The bounding circle approximate so assign each point to the closet polygon 
    sp_gdf[group_col] = sp_gdf['geometry'].apply(lambda x: sectors.distance(x).sort_values().index[0])

    # merge spatial points back onto orginal dataframe
    res_dissolved = sp_gdf.explode('gid_list')

    res_dissolved = res_dissolved[['gid_list', group_col]]

    merged_df = df.merge(
        res_dissolved,
        left_on='gid',
        right_on='gid_list',
        how='left',
        suffixes = [None, '_y']
    )



    return merged_df

    


def spatial_k_fold_generator(df, num_folds, group_col='group'):
    """ A wrapper so that spatial k fold can be used with the same syntax as sklearn k fold"""
    class _gen():
        def split(self, df_to_split):
            df_to_split = np.array(df_to_split)
            for k in range(num_folds):
                train_index = (df[group_col] != k)
                test_index = (df[group_col] == k)

                yield df_to_split[train_index], df_to_split[test_index]

    return _gen()



