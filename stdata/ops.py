import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import LineString, MultiPoint, Point
from sklearn.neighbors import BallTree

def _interp(xs, x1, x2, y1, y2) -> 'Point':
    def _intp(x, x1, x2, y1, y2):
        m = (y2-y1)/(x2-x1)
        return Point(x, m*(x-x1) + y1)
    
    return [_intp(x, x1, x2, y1, y2) for x in xs]

def discretise_linestring(linestring_geom: 'LineString', steps: int) -> 'MultiPoint':
    points = linestring_geom.coords
    num_points = len(points)
    all_points = []
    for p in range(num_points-1):
        x1,y1,_ = points[p]
        x2,y2,_ = points[p+1]
        x_spaced = np.linspace(x1, x2, steps)

        all_points = all_points + _interp(x_spaced, x1, x2, y1, y2)

    return MultiPoint(all_points)

# Taken directly from https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html

def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points

def ensure_continuous_timeseries(df, dt_col='date', id_col='id', freq='H', min_dt=None, max_dt = None):
    if min_dt is None:
        min_dt = df[dt_col].min()

    if max_dt is None:
        max_dt = df[dt_col].max()

    # Compute all datetimes between min_dt and max_dt with frequency freq
    all_dt = pd.DataFrame(
        pd.date_range(
            min_dt,
            max_dt,
            freq=freq
        ),
        columns=[dt_col]
    )

    all_sites = pd.DataFrame(pd.unique(df[id_col]), columns=[id_col])

    # construct full dateframe
    full_df = all_dt.merge(all_sites, how='cross')
    
    return full_df.merge(df, on=[dt_col, id_col], how='outer')
