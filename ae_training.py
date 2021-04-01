#!/usr/bin/env python3.8
# https://github.com/tensorflow/tensorflow/issues/47311
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import AEDense
from lib_VAE_outlier import input_handler
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
###############################################################################
# Loading training data
fname = f'spectra_{n_spectra}_{normalization_type}.npy'
fpath = f'{training_data_dir}/{fname}'

if os.path.exists(fpath):

    print(f'Loading training set: {fname}')

    training_set =  np.load(f'{fpath}')
    np.random.shuffle(training_set)

else:
    print(f'There is no file: {fname}')
    sys.exit()
###############################################################################
# Parameters for the AEDense
n_galaxies = training_set.shape[0]
n_input_dimensions = training_set[:, :-5].shape[1]
n_latent_dimensions = 10
###########################################
# encoder
n_layers_encoder = [600, 100]

# decoder
n_layers_decoder = [100, 600]

# Other parameters
# 1% to take advantage of stochastic part of stochastic gradient descent
batch_size = int(n_galaxies*0.01)
print(f'Batch size is: {batch_size}')

epochs = 5
learning_rate = 0.001 # default: 0.001
loss = 'mse'

ae = AEDense(n_input_dimensions, n_layers_encoder, n_latent_dimensions,
    n_layers_decoder, batch_size, epochs, learning_rate, loss)

ae.summary()
###############################################################################
# Training the model

history = ae.fit(spectra=training_set[:, :-5])
print(type(history))
print(history)
###############################################################################
# Defining directorie to save the model once it is trained
models_dir = f'{working_dir}/models/AE'

if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)
# Models names
ae_name = f'DenseAE_{loss}_{n_galaxies}'
encoder_name = f'DenseEncoder_{loss}_{n_galaxies}'
decoder_name = f'DenseDecoder_{loss}_{n_galaxies}'

if local:

    print('Saving model trained in local machine')
    ae.save_ae(f'{models_dir}/{ae_name}_{local}')
    ae.save_encoder(f'{models_dir}/{encoder_name}_{local}')
    ae.save_decoder(f'{models_dir}/{decoder_name}_{local}')

else:
    ae.save_ae(f'{models_dir}/{ae_name}')
    ae.save_encoder(f'{models_dir}/{encoder_name}')
    ae.save_decoder(f'{models_dir}/{decoder_name}')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
