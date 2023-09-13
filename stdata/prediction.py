""" Common functions that are usually required for runnign experiments. """
import numpy as np
from tqdm import tqdm, trange

def batch_predict(XS, prediction_fn=None, batch_size=1000, verbose=False, axis=0, ci=False, concat=True):
    # Ensure batch is less than the number of test points
    if XS.shape[0] < batch_size:
        batch_size = XS.shape[0]

    # Split up test points into equal batches
    num_batches = int(np.ceil(XS.shape[0] / batch_size))

    if ci:
        ys_median_arr = []
        ys_lower_arr = []
        ys_upper_arr = []
    else:
        ys_arr = []
        ys_var_arr = []
    index = 0

    if verbose:
        bar = tqdm(total=num_batches)

    for count in range(num_batches):
        if count == num_batches - 1:
            # in last batch just use remaining of test points
            batch = XS[index:, :]
        else:
            batch = XS[index : index + batch_size, :]

        index = index + batch_size

        # predict for current batch
        if ci:
            y_median, y_ci_lower, y_ci_upper = prediction_fn(batch)

            ys_median_arr.append(y_median)
            ys_lower_arr.append(y_ci_lower)
            ys_upper_arr.append(y_ci_upper)
        else:
            y_mean, y_var = prediction_fn(batch)
            ys_arr.append(y_mean)
            ys_var_arr.append(y_var)

        if verbose:
             bar.update(1)

    if ci:
        y_median = np.concatenate(ys_median_arr, axis=axis)
        y_ci_lower = np.concatenate(ys_lower_arr, axis=axis)
        y_ci_upper = np.concatenate(ys_upper_arr, axis=axis)

        return y_median, y_ci_lower, y_ci_upper
    else:
        if concat:
            y_mean = np.concatenate(ys_arr, axis=axis)
            try:
                y_var = np.concatenate(ys_var_arr, axis=axis)
            except:
                y_var = np.vstack(ys_var_arr)
        else:
            y_mean = ys_arr
            y_var = ys_var_arr


        return y_mean, y_var

def slice_array(A, arr):
    _A = []
    for a in arr:
        _A.append(A[a, :])
    return np.vstack(_A)

def slice_array_insert(A,B, arr):
    for i, a in enumerate(arr):
        _A_len = A[a].shape[0]

        A[a] = A[a] + B[i*_A_len:(i+1)*_A_len, :]

    return A


def st_batch_predict(model , XS, prediction_fn=None, batch_size=5000, verbose=False, out_dim=1, transpose_pred=False):
    """
        With KF models the prediction data must be at all timesteps. This function
            breaks up the space-time data into space-time batches (which spans across all time points)

        Args:
            batch_size: int - number of spatial points to process at a time
    """
    #sort XS into grid/'kronecker' structure

    XS_start = XS
    XS = np.roll(XS, -1, axis=1)
    #sort by time points
    grid_idx = np.lexsort(XS.T)
    #reset time axis
    XS = np.roll(XS, 1, axis=1)

    inv_grid_idx = np.argsort(grid_idx)

    XS = XS[grid_idx]


    time_points = np.unique(XS[:, 0])
    num_time_points = time_points.shape[0]
    num_spatial_points = int(XS.shape[0]/num_time_points)

    #number of spatial points that fit into batch

    #num_spatial_points_per_batch = int(np.floor(batch_size/num_time_points))
    num_spatial_points_per_batch = min(batch_size, num_spatial_points)

    num_steps = max(1, int(np.floor(num_spatial_points/num_spatial_points_per_batch)))

    if verbose:
        print('num_time_points: ', num_time_points)
        print('num_spatial_points: ', num_spatial_points)
        print('num_steps: ', num_steps)
        print('num_spatial_points_per_batch: ', num_spatial_points_per_batch)

    #empty prediction data
    mean = np.zeros([XS.shape[0], out_dim])
    var = np.zeros_like(mean)

    for i in trange(num_steps):

        batch = num_spatial_points_per_batch
        if i == num_steps-1:
            #select the remaining spatial points
            batch = num_spatial_points - i*num_spatial_points_per_batch

        #k*num_spatial_points is the index to the j'th time slice
        #i*num_spatial_points_per_batch is index to current spatial batch
        start_idx = lambda j: j*num_spatial_points + i*num_spatial_points_per_batch
        end_idx = lambda j: start_idx(j) + batch

        step_idx = [
            slice(start_idx(j), end_idx(j)) for j in range(num_time_points)
        ]

        _XS = slice_array(XS, step_idx)

        if prediction_fn is not None:
            _mean, _var = prediction_fn(_XS)
        else:
            _mean, _var = model.predict_y(_XS, diagonal_var=True)

        if transpose_pred:
            _mean = np.squeeze(_mean).T
            _var = np.squeeze(_var).T

        _mean = np.squeeze(_mean).reshape([-1, out_dim])
        _var = np.squeeze(_var).reshape([-1, out_dim])

        mean = slice_array_insert(mean, _mean, step_idx)
        var = slice_array_insert(var, _var, step_idx)

    #unsort grid/kronecker structre
    return [mean[inv_grid_idx]], [var[inv_grid_idx]]
