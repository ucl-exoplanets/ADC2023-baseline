import numpy as np

from helper import augment_data,standardise
def augment_data_with_noise(spectra, noise, repeat ):
    aug_spectra = augment_data(spectra, noise, repeat)
    aug_spectra = aug_spectra.reshape(-1, spectra.shape[1])
    return aug_spectra

def transform_data(org_arr, aug_arr=None):
    global_mean = np.mean(org_arr)
    global_std = np.std(org_arr)
    if aug_arr is not None:
        std_aug_spectra = standardise(aug_arr, global_mean, global_std)
    else:
        std_aug_spectra = standardise(org_arr, global_mean, global_std)
    std_aug_spectra = std_aug_spectra.reshape(-1, org_arr.shape[1])
    return std_aug_spectra, global_mean,global_std

