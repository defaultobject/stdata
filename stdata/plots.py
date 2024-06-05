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


def edited_mark_inset(fig, parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    # loc1, loc2 : {1, 2, 3, 4} 
    #'upper right'  : 1,
    #'upper left'   : 2,
    #'lower left'   : 3,
    #'lower right'  : 4
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
    import matplotlib.colors as mcolors
    import matplotlib.lines as lines
    import matplotlib.patches as mpatches

    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    

    print('inset_axes: ', inset_axes)
    print('inset_axes.viewLim: ', inset_axes.viewLim)
    print('points: ', pp.get_path())
    print('rect: ', rect.get_points())
    
    verts = pp.get_path().vertices
    trans = pp.get_patch_transform()
    points = trans.transform(verts)
    
    
    inax_position = inset_axes.transAxes.transform([0, 1]) #transform to display coords
    infig_position = inset_axes.figure.transFigure.inverted().transform(inax_position)  #transform to fig coords
    
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    #inset_axes.add_patch(p1)
    print('axes: ', parent_axes.transLimits.transform(inset_axes.viewLim))
    
    print('parent axis: ', parent_axes.get_aspect(), parent_axes.get_adjustable())
    print('parent axis ps original: ', parent_axes.get_position(original=True))
    print('parent axis ps: ', parent_axes.get_position())
    parent_axes.viewLim
    data_to_axes = parent_axes.transData.inverted() #transform to display coords
    axes_coords = parent_axes.transLimits.transform(parent_axes.viewLim)
    trans = parent_axes.figure.transFigure.inverted().transform(
        parent_axes.transAxes.transform(
            mpl.transforms.Affine2D().translate(0, -0.123).transform(axes_coords[0])
        )
        #axes_coords
    ) #transform to fig coords
    
    #trans = [[0.18185611530794643, 0.3329441062966627]]
    
    print('trans: ', trans)
    
    #print('p1: ', p1)
    #print('path: ', path.vertices)
    #p1.set_clip_on(False)
    #inset_axes.add_patch(p1)
    
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)

    
    if loc2a is not None:
        p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
        inset_axes.add_patch(p2)
        p2.set_clip_on(False)

    #return pp, p1, p2
    
def plot_zoom_in_bars(top_ax, bottom_ax):
    edited_mark_inset(
        None,
        top_ax,
        bottom_ax,
        2,
        3, 
        1,
        4
    )
