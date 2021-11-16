import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

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

    # works well for london
    # compute unique spatial points
    if verbose:
        print('Computing unique spatial points')


    spatial_points_df = points_flat_gdf.groupby([lat_col, lon_col]).agg({'gid': [len, list]}).reset_index()
    # flatten multi level index from groupby
    spatial_points_df.columns = [' '.join(col).strip().replace(' ', '_') for col in spatial_points_df.columns.values]

    sp_gdf = gpd.GeoDataFrame(
        spatial_points_df, 
        geometry=gpd.points_from_xy(spatial_points_df['lon'], spatial_points_df['lat'])
    )
    sp_gdf = sp_gdf.set_crs(epsg='4326')
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
        how='intersection'
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
