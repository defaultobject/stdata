import numpy as np

def create_spatial_grid(x1, x2, y1, y2, n1, n2):
    x = np.linspace(x1, x2, n1)
    y = np.linspace(y1, y2, n2)
    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])
    return np.array(grid)

def create_spatial_temporal_grid(time_points, x1, x2, y1, y2, n1, n2):
    x = np.linspace(x1, x2, n1)
    y = np.linspace(y1, y2, n2)
    grid = []
    for t in time_points:
        for i in x:
            for j in y:
                grid.append([t, i, j])
    return np.array(grid)

def create_geopandas_spatial_grid(xmin, xmax, ymin, ymax, cell_size_x, cell_size_y, crs=None):
    """
        see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    """
    import geopandas
    import shapely

    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax-cell_size_x, cell_size_x ):
        for y0 in np.arange(ymin, ymax-cell_size_y, cell_size_y):
            # bounds
            x1 = x0+cell_size_x
            y1 = y0+cell_size_y
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1)  )

    cell = geopandas.GeoDataFrame(grid_cells, columns=['geometry'], 
                                 crs=crs)
    return cell


def pad_with_nan_to_make_grid(X, Y):
    #converts data into grid

    N = X.shape[0]

    #construct target grid
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)

    Nt = unique_time.shape[0]
    Ns = unique_space.shape[0]

    print('grid size:', N, Nt, Ns, Nt*Ns)

    X_tmp = np.tile(np.expand_dims(unique_space, 0), [Nt, 1, 1])

    time_tmp = np.tile(unique_time, [Ns]).reshape([Nt, Ns], order='F')

    X_tmp = X_tmp.reshape([Nt*Ns, -1])

    time_tmp = time_tmp.reshape([Nt*Ns, 1])

    #X_tmp is the full grid
    X_tmp = np.hstack([time_tmp, X_tmp])

    #Find the indexes in X_tmp that we need to add to X to make a full grid
    _X = np.vstack([X,  X_tmp])
    _Y = np.nan*np.zeros([_X.shape[0], 1])

    _, idx = np.unique(_X, return_index=True, axis=0)
    idx = idx[idx>=N]
    print('unique points: ', idx.shape)

    X_to_add = _X[idx, :]
    Y_to_add = _Y[idx, :]

    X_grid = np.vstack([X, X_to_add])
    Y_grid = np.vstack([Y, Y_to_add])

    #sort for good measure
    _X = np.roll(X_grid, -1, axis=1)
    #sort by time points first
    idx = np.lexsort(_X.T)

    return X_grid[idx], Y_grid[idx]

