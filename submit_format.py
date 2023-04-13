import pandas as pd
import h5py
from tqdm import tqdm
import numpy as np 


def to_competition_format(tracedata_arr, weights_arr, name="submission.hdf5"):
    """convert input into competition format.
    we assume the test data is arranged in assending order of the planet ID.

    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 7, where M is the number of tracedata, here we assume tracedata is of equal size. It does not have to be but you will need to craete an alternative function if the size is different. 
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata

    Returns:
        None
    """
    submit_file = name
    RT_submission = h5py.File(submit_file,'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights_arr[n]),1)

        grp = RT_submission.create_group(f"Planet_public{n+1}")
        pl_id = grp.attrs['ID'] = n 
        tracedata = grp.create_dataset('tracedata',data=tracedata_arr[n])         
        weight_adjusted = weights_arr[n]

        weights = grp.create_dataset('weights',data=weight_adjusted)
    RT_submission.close()