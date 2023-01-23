""" Common functions that are usually required for runnign experiments. """
import numpy as np
from tqdm import tqdm

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

