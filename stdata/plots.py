import numpy as np
import pandas as pd

def grid_to_matrix(X, Y, x_col='lon', y_col='lat'):
    """
    X is a dataframe that holds a flattened N by D grid. Return Y in a matrix with corresponding extents
    Assumes that origin is lower left
    e.g.
        plt.imshow(Y_mat, extent=extents, origin='lower'
    """
    
    sort_idx = np.lexsort([X[x_col], X[y_col]])

    N1 = np.unique(X[x_col]).shape[0]
    N2 = np.unique(X[y_col]).shape[0]

    Y = np.array(Y)[sort_idx]
    Y = Y.reshape([N2, N1])

    extents = [
        np.min(X[x_col]), 
        np.max(X[x_col]), 
        np.min(X[y_col]), 
        np.max(X[y_col]), 
    ]

    return Y, extents

def plot_polygon_collection(ax, geoms, norm, values=None, colormap='Set1',  facecolor=None, edgecolor=None, alpha=1.0, linewidth=1.0, **kwargs):
    """ Plot a collection of Polygon geometries """

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    import shapely
    patches = []

    for poly in geoms:

        a = np.asarray(poly.exterior)
        if poly.has_z:
            poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))

        patches.append(Polygon(a))

    patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, norm=norm, **kwargs)

    if values is not None:
        patches.set_array(values)
        patches.set_cmap(colormap)


    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches
