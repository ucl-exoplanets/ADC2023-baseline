# ADC2023-baseline
Baseline solution for ADC2023 ECML competition

Inside this repo you will find the baseline solution for the Ariel Data Challenge. To run the script you will need access to the training and test data, both of which can be found from the [website](https://www.ariel-datachallenge.space/). 
There are two ways to run the baseline:

1. via command line:
```
python run_baseline.py --training PATH/TO/TRAININGDATA/ --test PATH/TO/TESTDATA
```

2. via jupyter notebook (also contains the evaluation metric), ADC2023-baseline.ipynb

## Description of the network
We trained a neural network to perform a supervised multi-target regression task. The architecture of the network is modified from the CNN network as described in [Yip et al.](https://arxiv.org/abs/2011.11284). 

## Preprocessing Steps
- We used the first 5000 data instances to train the model
- We augmented the data with the observation noise
- Used stellar and planetary radii as additional features
- Standardised both inputs and output

At test time we performed [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to provide a mutlivariate distribution for each test example. It is then packaged ready for submission

## Metrics
We have inlcuded the algorithm we used to compute the metric score in the jupyter notebook. Note that in order to compute the final score, you must have taurex3 installed in your python environment, with the appropriate linelist, which can be found [here](https://www.dropbox.com/sh/1njwmcqvv8zj3sy/AABenL2JtAh6QrTwTPfxnTjya?dl=0). For installation procedure of taurex3, please see [here](https://github.com/ucl-exoplanets/TauREx3_public)

For more information about the metric please see here.

## Things to improve
There are different direction to take from here on, let us summarise the shortcoming of this model:
- The data preprocessing is quite simplitic and could have invested more efforts.
- we have only used 5000 data points, instead of the full dataset
- we didnt train the model with results from the retrieval (Tracedata.hdf5), which are the GT for this competition.
- The conditional distribution from MCDropout is very restricted and Gaussian-like
- So far we havent considered the atmospheric targets as a joint distribution
- We have only used stellar radius from the auxillary information
- We have not done any hyperparameter tuning 


