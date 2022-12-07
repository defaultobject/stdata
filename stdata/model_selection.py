import pandas as pd
import numpy as np

import geopandas as gpd
from geopandas import GeoSeries

from .feature_extraction import _get_unique_spatial_points
import sklearn
from sklearn import cluster
from sklearn.metrics import pairwise_distances

from tqdm import tqdm


def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to, ignore_nan=False):
    if ignore_nan:
        mean_fn = np.nanmean
        std_fn = np.nanstd
    else:
        mean_fn = np.mean
        std_fn = np.std

    return (x - mean_fn(wrt_to, axis=0))/std_fn(wrt_to, axis=0)

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
    from shapely.geometry import Polygon, Point


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

def _equal_k_means(df, n_clusters=5, verbose=False):
    """
    This uses a greedy algorithm, and so although each cluster will be equal, it may not be spatially aligned.
    """
    df = df.copy().reset_index()
    
    df['__index'] = df.index
    
    X = np.array(df[['lat', 'lon']])
    
    m = cluster.KMeans(n_clusters=n_clusters).fit(X)
    
    dists = pairwise_distances(m.cluster_centers_, X)
    
    clusters = {c: [] for c in range(n_clusters)}
    
    assigned_ids = []

    N = X.shape[0]

    dists_all = [dists[c].argsort() for c in range(n_clusters)]
    
    # each step assigns n_cluster points to assigned_ids
    num_iters = int(np.ceil(N/float(n_clusters)))

    if verbose:
        bar = tqdm(total=num_iters)

    for i in range(num_iters):
        for c in range(n_clusters):

            # find closest point
            all_closest_points = dists_all[c]
            
            closest_points = all_closest_points[~np.isin(all_closest_points,assigned_ids)]
            closest_point = closest_points[0]
            closest_point_dist = dists[c][closest_point]
            
            # find the closest cluster for closest point
            closest_cluster = dists[:, closest_point].argsort()[0]
            
            if c != closest_cluster:
                # find assigned point in cluster that is closest to c
                closest_points_in_new_cluster = dists[c][
                    clusters[closest_cluster]
                ].argsort()
                
                if len(closest_points_in_new_cluster) == 0:
                    clusters[c].append(closest_point)
                else:
                    closest_point_in_new_cluster = clusters[closest_cluster][closest_points_in_new_cluster[0]]

                    if dists[c][closest_point_in_new_cluster] < closest_point_dist:
                        clusters[closest_cluster].remove(closest_point_in_new_cluster)
                        clusters[closest_cluster].append(closest_point)
                        clusters[c].append(closest_point_in_new_cluster)
                    else:
                        # do nothing
                        clusters[c].append(closest_point)
            else:
                clusters[c].append(closest_point)

            assigned_ids.append(closest_point)
            

            if len(assigned_ids) == N:
                break
                
        if len(assigned_ids) == N:
            break

        if verbose:
         bar.update(1)
            
    cluster_df = pd.DataFrame(
        [[i, c] for c, a in clusters.items() for i in a], 
        columns=['__index_cluster', 'label']
    )
    
    df = df.merge(cluster_df, left_on=['__index'], right_on=['__index_cluster'], how='left', suffixes=[None, '_y'])
    df = df.drop(columns=['__index', '__index_cluster'])

    df['k_means_label'] = m.labels_    
    
    return df

def equal_spatial_clusters(df, n_clusters=5, lat_col='lat', lon_col='lon', group_col='label', verbose=False):
    df = df.copy().reset_index()

    #assign gid so we can avoid spatial merge later
    df['gid'] = df.index

    sp_gdf = _get_unique_spatial_points(df, lat_col, lon_col)

    sp_gdf = _equal_k_means(sp_gdf, n_clusters, verbose=verbose)

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


def equal_spatial_clusters_with_support(df, support_df, lat_col='lat', lon_col='lon', group_col='label', buffer_size=100, n_clusters=5, verbose=True):
    df = df.copy()
    support_df = support_df.copy()

    df_sites = _get_unique_spatial_points(df, lat_col, lon_col)
    support_sites = _get_unique_spatial_points(support_df, lat_col, lon_col)

    df_sites = df_sites.to_crs('EPSG:27700')
    support_sites = support_sites.to_crs('EPSG:27700')

    if verbose:
        print(f'{df.shape} -> {df_sites.shape} sites')
        print(f'{support_df.shape} -> {support_sites.shape} sites')

    # we call _get_unique_spatial_points multiple times so rename to avoid merging errors
    df_sites = df_sites.rename(columns={'gid_list': 'outer_gid_list', 'gid_len': 'outer_gid_len'})
    support_sites = support_sites.rename(columns={'gid_list': 'outer_gid_list', 'gid_len': 'outer_gid_len'})

    #convert to flat crs
    df_sites = df_sites.to_crs('EPSG:27700')
    support_sites = support_sites.to_crs('EPSG:27700')

    # add site id to keep track of points
    df_sites['site_id'] = df_sites.index
    support_sites['site_id'] = support_sites.index

    # construct spatial support buffers
    support_buffers = gpd.GeoDataFrame(
        {'geometry': support_sites.buffer(buffer_size)}
    )

    # get df sites within the spatial support region
    res_join = gpd.sjoin(
        support_buffers, df_sites, how='inner', op='intersects'
    )

    _, unique_idx = np.unique(res_join['site_id'], return_index=True)

    res_join = res_join.iloc[unique_idx]

    # get equal buffers

    res_join_new = equal_spatial_clusters(res_join, n_clusters)

    # undo rename so we can merge sites back onto original df
    res_join_new = res_join_new.drop(columns=['gid', 'gid_list'])
    res_join_new = res_join_new.rename(columns={'outer_gid_len': 'gid_len', 'outer_gid_list': 'gid_list'})

    res_dissolved = res_join_new.explode('gid_list')

    merged_df = df.merge(
        res_dissolved,
        left_on='gid',
        right_on='gid_list',
        how='left',
        suffixes = [None, '_y']
    )

    merged_df[group_col] = merged_df[group_col].fillna(-1)

    return merged_df


def train_test_split_with_only_one_missing_task(split_percent, X, Y, seed=None):
    """
    Train-test split support for multi-tasks.

    We want to measure the performance of when we have AT LEAST one output
       so we first get a random choice of N * (10 * P / 100), because we are treating them randomly
    """ 
    #Y must be a float to support nans
    Y = np.copy(Y).astype(float)
    X = np.copy(X)

    N, P = Y.shape
    N_across_all_groups = int(N*(split_percent * P / 100))

    if N < N_across_all_groups:
        raise RuntimeError('There are too many tasks, try lowering the split percent?')

    train_indexes = list(range(N))

    if seed is not None:
        np.random.seed(seed)

    test_indexes = np.random.choice(range(N), N_across_all_groups, replace=False)

    # split into sets of disjoint <split_percent> testing sets
    # This forces equal sized groupes
    # i.e not all of the N_across_all_groups may be used
    test_index_list = [
        test_indexes[int(N_across_all_groups/P) * i: int(N_across_all_groups/P) * (i + 1)]
        for i in range(P)
    ]
    # should be unique
    assert np.unique(test_index_list).shape[0] == np.prod(np.array(test_index_list).shape)

    # construct training indexes

    train_index_list = [
        list(set(train_indexes) - set(test_index_list[i])) 
        for i in range(P) 
    ]

    # construct train sets
    Y_train = Y.copy()
    for p in range(P):
        Y_train[test_index_list[p], p] = np.NaN

    # should only be one nan in each row
    assert np.any(np.sum(np.isnan(Y), axis=1) > 1) == False

    # Construct testing sets
    Y_test = Y.copy()
    X_test = X.copy()

    # reverse the setting of NaNs as now we want ONLY want to validate p
    for p in range(P):
        col_index_without_p = list(set(range(P)) - set([p]))
        Y_test[np.ix_(test_index_list[p], col_index_without_p)] = np.NaN

    # for testing only select colums with nans 
    data_test_index = np.any(np.isnan(Y_train), axis=1)

    # only select testing locations
    X_test = X[data_test_index].copy()
    Y_test = Y_test[data_test_index]

    return X, Y_train, X_test, Y_test, data_test_index

