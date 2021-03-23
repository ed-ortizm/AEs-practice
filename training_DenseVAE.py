#!/usr/bin/env python3.8

import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from constants_VAE_outlier import normalization_schemes
from constants_VAE_outlier import spectra_path
from lib_VAE_outlier import DenseEncoder, DenseDecoder, DenseVAE
###############################################################################
ti = time.time()
###############################################################################
local = sys.argv[1]=='local'

if local:
    print('We are in local')
    n_spectra = 1_000
else:
    print('We are in remote')
    n_spectra = int(sys.argv[2])

if sys.argv[3] in normalization_schemes:

    normalization_type = sys.argv[3]

    print(f'normalization type: {normalization_type}')

else:
    print('Normalyzation type should be: median, min_max or Z')
    sys.exit()

###############################################################################
# Relevant directories
training_data_dir = f'{spectra_path}/normalized_data'
###############################################################################
# Loading training data

fname = f'spectra_{n_spectra}_{normalization_type}.npy'
fpath = f'{training_data_dir}/{fname}'

if os.path.exists(fpath):

    print(f'Loading training set: {fname}')

    training_set =  np.load(f'{training_data_dir}/{fname}', mmap_mode='r')

else:
    print(f'There is no file: {fname}')
###############################################################################
# Parameters for the DenseVAE
n_latent_dimensions = 5
###########################################
# encoder
n_input_dimensions = training_set.shape[1]
n_layers_encoder = [100, 50, 20]

encoder = DenseEncoder(n_input_dimensions=n_input_dimensions,
    n_hiden_layers=n_layers_encoder, n_latent_dimensions=n_latent_dimensions
)
###########################################
# decoder
n_layers_decoder = [20, 50, 100]

decoder = DenseDecoder(n_latent_dimensions=n_latent_dimensions,
    n_output_dimensions=n_input_dimensions, n_hiden_layers=n_layers_decoder
)
###########################################
# vae
vae = DenseVAE(encoder=encoder, decoder=decoder)
vae.summary()
###############################################################################
###############################################################################
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
