import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.strtree import STRtree
from .ops import discretise_linestring, nearest_neighbor

def _get_unique_spatial_points(points_gdf, lat_col, lon_col):
    spatial_points_df = points_gdf.groupby([lat_col, lon_col]).agg({'gid': [len, list]}).reset_index()
    # flatten multi level index from groupby
    spatial_points_df.columns = [' '.join(col).strip().replace(' ', '_') for col in spatial_points_df.columns.values]

    # TODO: just use point geometry not lat lon
    sp_gdf = gpd.GeoDataFrame(
        spatial_points_df, 
        geometry=gpd.points_from_xy(spatial_points_df[lon_col], spatial_points_df[lat_col])
    )

    sp_gdf = sp_gdf.set_crs(epsg='4326')

    return sp_gdf

def static_approx_distance_linestring(points_gdf, map_gdf, flat_crs = 27700, verbose=False, map_tree = None, convert_to_flat=True, target_col='distance'):
    discretize_size = 10
    lat_col = 'lat'
    lon_col = 'lon'

    points_gdf = points_gdf.copy()
    map_gdf = map_gdf.copy()

    #assign gid so we can avoid spatial merge later
    points_gdf['gid'] = points_gdf.index


    # compute unique spatial points
    if verbose:
        print('Computing unique spatial points')


    # points will be retured in 4326
    sp_gdf = _get_unique_spatial_points(points_gdf, lat_col, lon_col)

    # store geometry so it does not get overwritten

    map_gdf['__geometry'] = map_gdf['geometry']

    # convert geometry from linestring to points by dicreitising the lines

    map_gdf['split_geom'] = map_gdf['geometry'].apply(
        lambda geom: discretise_linestring(geom, discretize_size)
    )


    # Make the discretised geom the active geometry
    map_gdf['geometry'] = map_gdf['split_geom']

    # Convert multi point to point
    exploded_map_gdf = map_gdf.explode()

    # compute closest dist from the spatial points to the exploded points
    res_df = nearest_neighbor(sp_gdf, exploded_map_gdf, return_dist=True)

    sp_gdf[target_col] = res_df['distance']

    # merge back onto the original dataframe

    res_dissolved = sp_gdf.explode('gid_list')

    merged_df = points_gdf.merge(
        res_dissolved,
        left_on='gid',
        right_on='gid_list',
        how='left',
        suffixes = [None, '_y']
    )

    return merged_df



def static_feature(points_gdf, map_gdf, buffer_size=100, flat_crs = 27700, verbose=False):
    """
    Extract features from map_gdf for each spatial point in points_gdf. 
    This done by computing a buffer around each point and finding the intersection of these with map_gdf.
    From this statistics like total area, total length, etc can be computed
    """
    #assign gid so we can avoid spatial merge later
    points_gdf['gid'] = points_gdf.index

    lat_col = 'lat'
    lon_col = 'lon'

    points_crs = points_gdf.crs
    map_crs = map_gdf.crs

    # convert to crs 27700 so we can work in meters 
    # epsg:27700 is good for England

    if verbose:
        print('Converting to flat crs')

    points_flat_gdf = points_gdf.to_crs({'init': f'EPSG:{flat_crs}'})
    map_flat_gdf = map_gdf.to_crs({'init': f'EPSG:{flat_crs}'})

    # compute unique spatial points
    if verbose:
        print('Computing unique spatial points')


    sp_gdf = _get_unique_spatial_points(points_flat_gdf, lat_col, lon_col)
    sp_gdf = sp_gdf.to_crs(epsg=flat_crs)


    # construct buffer around each spatial point
    if verbose:
        print('Computing buffers')

    sp_gdf['buffer_geom'] = sp_gdf.buffer(buffer_size)
    # add index so we merge back on later
    sp_gdf['id'] = sp_gdf.index

    # make buffer the active geom
    sp_gdf['point_geom'] = sp_gdf['geometry']
    sp_gdf['geometry'] = sp_gdf['buffer_geom']

    # intersect buffer with map
    if verbose:
        print('Intersecting buffer and map')

    res = gpd.overlay(
        map_flat_gdf, 
        sp_gdf, 
        how='intersection',
        make_valid=True
    )
    #store geometry so that it does not get discarded after sjoin
    res['right_geom'] = res.geometry

    # merge intersection back onto the spatial points
    if verbose:
        print('Rejoining back onto spatial points')

    res_join = gpd.sjoin(sp_gdf, res, how='left', op='intersects')

    # make sure that the id column exists
    res_join = res_join.rename(columns={'id_left':'id'})

    # make the map geom the active geometry
    res_join['left_geometry'] = res_join['geometry']
    res_join['geometry'] = res_join['right_geom']

    # remove nan geoms (happens when there is no intersection)
    res_join = res_join[~pd.isna(res_join['geometry'])]

    # group by spatial point id
    if verbose:
        print('Dissolving')
        #breakpoint()


    # Ensure geometries are valid
    res_join['geometry'] = res_join.buffer(0.01)

    res_dissolved = res_join.dissolve(by='id').reset_index()

    res_dissolved = res_dissolved.rename(
        columns={
            'gid_list_left':'gid_list',
            'buffer_geom_left':'buffer_geom',
        }
    )

    res_dissolved = res_dissolved[['id', 'gid_list', 'buffer_geom', 'geometry']]

    # compute statistics
    if verbose:
        print('Computing statistics')

    res_dissolved['length']  = res_dissolved.geometry.length

    # merge back onto the original dataframe

    res_dissolved = res_dissolved.explode('gid_list')

    merged_df = points_gdf.merge(
        res_dissolved,
        left_on='gid',
        right_on='gid_list',
        how='left',
        suffixes = [None, '_y']
    )

    return merged_df
