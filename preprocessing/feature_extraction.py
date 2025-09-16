
# will load audio files, extract wavenet and MFCC features and save them in data/features/sequnetial
# features for each audio file in the GTZAN dataset
#%%
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
from IPython.display import Audio

import tensorflow as tf
import os
import librosa

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import logging
logging.getLogger('tensorflow').disabled = True
tf.config.threading.set_inter_op_parallelism_threads(7)
tf.config.threading.set_intra_op_parallelism_threads(7)
import os
def list_folders(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]


def list_files(directory):
    return [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]

#%%

import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    try:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU(s) is/are available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected by TensorFlow. Encoding will run on CPU.")

#%%
genres = list_folders('../data/audio/GTZAN/genres')
checkpoint_path = '../wavenet-ckpt/model.ckpt-200000'



for i in range(len(genres)):
    wav_files = list_files('../data/audio//GTZAN/genres/' + genres[i])
    wav_files = [s for s in wav_files if s.find('_') == -1]

    for wav_file in wav_files:
        fname = '../data/audio/GTZAN/genres/' + genres[i] + '/' + wav_file
        print(fname)

        #wavnet feature extraction
        MODEL_SR = 16000
        duration_seconds = librosa.get_duration(filename=fname)
        desired_sample_length_at_model_sr = int(duration_seconds * MODEL_SR)
        audio_data = utils.load_audio(fname,
                                      sample_length=desired_sample_length_at_model_sr,  # Now passes 480,000
                                      sr=MODEL_SR)

        sample_length = audio_data.shape[0]
        #encoding = fastgen.encode(audio_data,
                                  #checkpoint_path=checkpoint_path,
                                  #sample_length=sample_length)

        saveplace = '../data/features/sequential/wavenet/' + genres[i] + '/' + wav_file[:-4]
        os.makedirs('../data/features/sequential/wavenet/' + genres[i], exist_ok=True)
        print('saveplace', saveplace)
        #np.save(saveplace + '.npy', encoding)

        #mfcc feature extraction
        sr = librosa.get_samplerate(fname)
        sample_rate = sr

        data, _ = librosa.load(fname)
        mfcc_size = 13
        trimmed_data, _ = librosa.effects.trim(y=data)

        mfccs = librosa.feature.mfcc(y=trimmed_data,
                                     sr=sample_rate,
                                     n_mfcc=mfcc_size)
        saveplace_mfcc = '../data/features/sequential/mfcc/' + genres[i] + '/' + wav_file[:-4]
        os.makedirs('../data/features/sequential/mfcc/' + genres[i], exist_ok=True)
        print('saveplace', saveplace_mfcc)
        np.save(saveplace_mfcc + '.npy', mfccs) 
        



