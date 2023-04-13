import numpy as np


def to_observed_matrix(data_file,aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum


def standardise(arr, mean, std):
    return (arr-mean)/std

def transform_back(arr, mean, std):
    return arr*std+mean

def augment_data(arr, noise, repeat=10):
    noise_profile = np.random.normal(loc=0, scale=noise, size=(repeat,arr.shape[0], arr.shape[1]))
    ## produce noised version of the spectra
    aug_arr = arr[np.newaxis, ...] + noise_profile
    return aug_arr

def visualise_spectrum(spectrum):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,6))
    plt.errorbar(x=spectrum[:,0], y= spectrum[:,1], yerr=spectrum[:,2] )
    ## usually we visualise it in log-scale
    plt.xscale('log')
    plt.show()
    
def transform_and_reshape( y_pred_valid,targets_mean, targets_std,instances,N_testdata):
    y_pred_valid_org = transform_back(y_pred_valid,targets_mean[None, ...], targets_std[None, ...])
    y_pred_valid_org = y_pred_valid_org.reshape(instances, N_testdata, len(targets_std))
    y_pred_valid_org = np.swapaxes(y_pred_valid_org, 1,0)
    return y_pred_valid_org