#!/usr/bin/env python3.8
# https://github.com/tensorflow/tensorflow/issues/47311
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from constants_VAE_outlier import normalization_schemes
from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import DenseVAE
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
training_data_dir = f'{spectra_dir}/normalized_data'
###############################################################################
# Loading training data
fname = f'spectra_{n_spectra}_{normalization_type}.npy'
fpath = f'{training_data_dir}/{fname}'

if os.path.exists(fpath):

    print(f'Loading training set: {fname}')

    training_set =  np.load(f'{fpath}', mmap_mode='r')

else:
    print(f'There is no file: {fname}')
    sys.exit()
###############################################################################
# Parameters for the DenseVAE
n_input_dimensions = training_set[:, :-5].shape[1]
n_latent_dimensions = 5
###########################################
# encoder
n_layers_encoder = [100, 50]

# decoder
n_layers_decoder = [50, 100]

# DenseVAEv2
vae = DenseVAE(n_input_dimensions, n_layers_encoder, n_latent_dimensions,
    n_layers_decoder)

vae.vae.summary()
###############################################################################
# Training the model

vae.vae.fit(x=training_set[:, :-5], y=training_set[:, :-5], batch_size=10, epochs=20)
# ###############################################################################
# # Defining directorie to save the model once it is trained
# models_dir = f'{working_dir}/models'
#
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir, exist_ok=True)
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
