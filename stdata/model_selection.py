import pandas as pd
import numpy as np

def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to):
    return (x - np.mean(wrt_to, axis=0))/np.std(wrt_to, axis=0)

def un_normalise_df(x, wrt_to):
    return x* np.std(wrt_to, axis=0) + np.mean(wrt_to, axis=0)

def train_test_split_indices(N, split=0.5, seed=0):
    """ Compute a random split based on seed """

    np.random.seed(seed)
    rand_index = np.random.permutation(N)

    N_tr =  int(N * split) 

    return rand_index[:N_tr], rand_index[N_tr:] 

