import numpy as np

def read_data(path):
    import h5py
    trace = h5py.File(path,"r")
    return trace


def to_matrix(data_file_gt, predicted_file, aux_df):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    id_order = aux_df.index
    num = len(data_file_gt.keys())
    list_trace_gt = []
    list_weight_gt = []
    list_trace_predicted = []
    list_weight_predicted = []

    for x in id_order:
        current_planet_id = f'Planet_{x}'
        trace_gt_planet = np.array(data_file_gt[current_planet_id]['tracedata'])
        trace_weight_planet = np.array(data_file_gt[current_planet_id]['weights'])
        list_trace_gt.append(trace_gt_planet)
        list_weight_gt.append(trace_weight_planet)
        predicted_planet = np.array(predicted_file[current_planet_id]['tracedata'])
        predicted_weights = np.array(predicted_file[current_planet_id]['weights'])
        list_trace_predicted.append(predicted_planet)
        list_weight_predicted.append(predicted_weights)
        
    return list_trace_gt, list_weight_gt, list_trace_predicted,list_weight_predicted

def default_prior_bounds():
    """Prior bounds of each different molecules."""

    #### check here!!!!!!####
    Rp_range = [0.1, 3]
    T_range = [0,7000]
    gas1_range = [-12, -1]
    gas2_range = [-12, -1]
    gas3_range = [-12, -1]
    gas4_range = [-12, -1]
    gas5_range = [-12, -1]
    
    bounds_matrix = np.vstack([Rp_range,T_range,gas1_range,gas2_range,gas3_range,gas4_range,gas5_range])
    return bounds_matrix

def restrict_to_prior(arr, bounds_matrix):
    """Restrict any values within the array to the bounds given by a bounds_matrix.

    Args:
        arr (array): N-D array 
        bounds_matrix (array): an (N, 2) shaped matrix containing the min and max bounds , where N is the number of dimensions

    Returns:
        array: array with extremal values clipped. 
    """
    arr = np.clip(arr, bounds_matrix[:,0],bounds_matrix[:,1])
    return arr

def normalise_arr(arr, bounds_matrix, restrict = True):
    if restrict:
        arr = restrict_to_prior(arr, bounds_matrix)
    norm_arr = (arr - bounds_matrix[:,0])/(bounds_matrix[:,1]- bounds_matrix[:,0])
    return norm_arr

def preprocess_trace_for_posterior_loss(tr, weights, bounds):
    import nestle
    trace_resampled = nestle.resample_equal(tr,weights )
    trace = normalise_arr(trace_resampled, bounds )
    return trace 


def compute_posterior_loss(tr1, weight1, tr2, weight2, bounds_matrix=None):
    from scipy import stats
    if bounds_matrix is None:
        bounds_matrix = default_prior_bounds()
    n_targets = tr1.shape[1]
    trace1 = preprocess_trace_for_posterior_loss(tr1, weight1, bounds_matrix)
    trace2 = preprocess_trace_for_posterior_loss(tr2, weight2, bounds_matrix)

    score_trace = []
    for t in range(0, n_targets):
        resampled_gt = np.resize(trace2[:,t], trace1[:,t].shape)
        metric_ks = stats.ks_2samp(trace1[:,t], resampled_gt)
        score_trace.append((1 - metric_ks.statistic) * 1000)
    return np.array(score_trace).mean()