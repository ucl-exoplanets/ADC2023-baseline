import numpy as np
from posterior_utils import default_prior_bounds,restrict_to_prior
from FM_utils_final import *

def L2_loss(truth, predicted):
    """Simple MSE"""
    return np.mean(np.square(truth-predicted))
def L1_loss(truth, predicted):
    """Simple MAE"""
    return np.mean(np.abs(truth-predicted))

def huber_loss(truth, predicted, alpha):
    """huber loss with threshold (alpha) set at 1"""
    if alpha >= 1:  
        return L2_loss(truth, predicted)
    else:
        return L1_loss(truth, predicted)
    
def compute_score(median, bound, GT_median, GT_bound):
    """compute the score contribution from the similaries between two spectra.

    Args:
        median (array): median spectra from participants
        bound (array): The IQR bound from participants. 
        GT_median (array): median spectra generated from GT
        GT_bound (array): The IQR bound from GT.

    Returns:
        scalar: the score from spectral loss
    """
    GT_level = np.mean(GT_median)
    level = np.mean(median)
    alpha = np.abs(np.log10(level/GT_level))
    log_truth = np.log10(GT_median)
    log_predicted = np.log10(median)
    median_loss = 100*huber_loss(log_truth,log_predicted,alpha)
    log_bound = np.log10(bound)
    log_GTbound = np.log10(GT_bound)
    mean_bound = np.mean(bound)
    mean_GTbound = np.mean(GT_bound) 
    alpha_bound = np.abs(np.log10(mean_bound/mean_GTbound))
    bound_loss = 100*huber_loss(log_GTbound, log_bound,alpha_bound)
    score = 1000-np.mean([bound_loss,median_loss])
    ## the minimum score is 0 
    score = np.maximum(score, 0)
    return score

def compute_spectral_loss(tr1, weights1, tr2,weights2,bounds_matrix,fm_func,q_list):
    tr1 = restrict_to_prior(tr1, bounds_matrix)
    q1, q2, q3 = compute_approx_mean_and_bound(tr1, weights1, fm_func, q_list)
    q1, q2, q3 = check_output(q1, q2, q3)
    median, bound = q2, q3 - q1 + 1e-8

    ## compute for ground truth
    tr2 = restrict_to_prior(tr2, bounds_matrix)
    q1_GT, q2_GT, q3_GT = compute_approx_mean_and_bound(tr2, weights2, fm_func, q_list)
    q1_GT, q2_GT, q3_GT  = check_output(q1_GT, q2_GT, q3_GT)
    GT_median, GT_bound = q2_GT, q3_GT - q1_GT + 1e-8

    score = compute_score(median, bound, GT_median, GT_bound)
    return score