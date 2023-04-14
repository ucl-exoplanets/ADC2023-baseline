import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import h5py
import os
from tqdm import tqdm
from helper import *
from preprocessing import *
from submit_format import to_competition_format
from MCDropout import MC_Convtrainer
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument( "--training",
                    help="training path")
parser.add_argument( "--test",
                    help="test path")

args = parser.parse_args()


#### Global ###
SEED=42
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000
### Preprocessing setup ###
repeat = 5
threshold = 0.8 ## for train valid split.
N = 5000 # train on the first 5000 data instances, remember only some examples are labelled, others are unlabelled!
### Network setup ###
batch_size= 32
lr= 1e-3
epochs = 30
filters = [32,64,64]
dropout = 0.1
# number of examples to generate in evaluation time (5000 is max for this competition)
N_samples = 5000

#### Load Data ####
training_path = args.training
test_path = args.test

training_GT_path = os.path.join(training_path, 'Ground Truth Package')
spectral_training_data = h5py.File(os.path.join(training_path,'SpectralData.hdf5'),"r")
aux_training_data = pd.read_csv(os.path.join(training_path,'AuxillaryTable.csv'))
soft_label_data = pd.read_csv(os.path.join(training_GT_path, 'FM_Parameter_Table.csv'))

spec_matrix = to_observed_matrix(spectral_training_data,aux_training_data)

####### Preprocessing ########
## extract the noise
noise = spec_matrix[:N,:,2]
## We will incorporate the noise profile into the observed spectrum by treating the noise as Gaussian noise.
spectra = spec_matrix[:N,:,1]
wl_channels = len(spec_matrix[0,:,0])
global_mean = np.mean(spectra)
global_std = np.std(spectra)

Rs = aux_training_data[['star_radius_m',]]
## we would prefer to use Rsol
Rs['star_radius'] = Rs['star_radius_m']/RSOL
Rs = Rs.drop(['star_radius_m'],axis=1)
Rs = Rs.iloc[:N, :]
mean_Rs = Rs.mean()
stdev_Rs = Rs.std()

# 
target_labels = ['planet_radius','planet_temp','log_H2O','log_CO2','log_CO','log_CH4','log_NH3']
targets = soft_label_data.iloc[:N][target_labels]
num_targets = targets.shape[1]
targets_mean = targets.mean()
targets_std = targets.std()


#### Train Test Split ####
ind = np.random.rand(len(spectra)) < threshold
training_spectra, training_Rs,training_targets, training_noise = spectra[ind],Rs[ind],targets[ind], noise[ind]
valid_spectra, valid_Rs, valid_targets = spectra[~ind],Rs[~ind],targets[~ind]

#### augment data  ####
aug_spectra = augment_data_with_noise(training_spectra, training_noise, repeat)
aug_Rs = np.tile(training_Rs.values,(repeat,1))
aug_targets = np.tile(training_targets.values,(repeat,1))

#### standardise #####
std_aug_spectra = standardise(aug_spectra, global_mean, global_std)
std_aug_spectra = std_aug_spectra.reshape(-1, wl_channels)
std_valid_spectra = standardise(valid_spectra, global_mean, global_std)
std_valid_spectra = std_valid_spectra.reshape(-1, wl_channels)

std_aug_Rs= standardise(aug_Rs, mean_Rs.values.reshape(1,-1), stdev_Rs.values.reshape(1,-1))
std_valid_Rs= standardise(valid_Rs, mean_Rs, stdev_Rs)

std_aug_targets = standardise(aug_targets, targets_mean.values.reshape(1,-1), targets_std.values.reshape(1,-1))
std_valid_targets = standardise(valid_targets, targets_mean, targets_std)

##### Setup and run #####
model = MC_Convtrainer(wl_channels,num_targets,dropout,filters)

model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss='mse',)
model.fit([std_aug_spectra,std_aug_Rs], 
          std_aug_targets, 
          validation_data=([std_valid_spectra, std_valid_Rs],std_valid_targets),
          batch_size=batch_size, 
          epochs=epochs, 
          shuffle=False,)

### generate submission for leaderboard #### 
spec_test_data = h5py.File(os.path.join(test_path,'SpectralData.hdf5'),"r")
aux_test_data = pd.read_csv(os.path.join(test_path,'AuxillaryTable.csv'))

#### Preprocess in the same way as above ####
test_spec_matrix = to_observed_matrix(spec_test_data,aux_test_data )
std_test_spectra = standardise(test_spec_matrix[:,:,1], global_mean, global_std)

test_Rs = aux_test_data[['star_radius_m']]

test_Rs['star_radius'] = test_Rs['star_radius_m']/RSOL
test_Rs = test_Rs.drop(['star_radius_m'],axis=1)
std_test_Rs= standardise(test_Rs, mean_Rs, stdev_Rs)

tf.keras.utils.set_random_seed(SEED)

##### sample trace #####
instances = N_samples
y_pred_distribution = np.zeros((instances, len(std_test_spectra), num_targets ))
for i in tqdm(range(instances)):
    
    y_pred = model([std_test_spectra,std_test_Rs],training=True)
    y_pred_distribution[i] += y_pred

y_pred_distribution = y_pred_distribution.reshape(-1,num_targets)
y_pred_test_org = transform_and_reshape(y_pred_distribution,targets_mean, targets_std,instances,N_testdata=len(std_test_spectra))

## put it into competition format
tracedata = y_pred_test_org
# weight takes into account the importance of each point in the tracedata. 
weight = np.ones((tracedata.shape[0],tracedata.shape[1]))/np.sum(np.ones(tracedata.shape[1]) )

submission = to_competition_format(tracedata, 
                                        weight, 
                                        name="submission.hdf5")