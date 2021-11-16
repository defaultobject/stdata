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
