import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import numpy as np
import glob

original_fs = 496 # original sample rate
target_fs = 100 # new sample rate
nyquist_freq = target_fs / 2
order = 2
b, a = signal.butter(order, nyquist_freq, fs=original_fs, btype='lowpass')

filepath = "/data1/neurdylab/datasets/eegfmri_vu/PROC/physio/{filename}"
files = []
with open("data/pulseimpute_data/waveforms/eegfmri_vu_ppg/eegfmri_vu_ppg_good_files.txt") as text_file:
    for line in text_file:
        files.append(filepath.replace("{filename}", line.strip()))

for index, file in enumerate(files):
    mat = sio.loadmat(file)
    data = mat['OUT_p'][0]['card_dat'][0]
    
    data = data.flatten()
    data = signal.filtfilt(b, a, data) # nyquist lowpass filter
    
    data = data[0:round(30000*original_fs/target_fs)]
    data = signal.resample(data, 30000)
    
    data = data.reshape(data.shape[0],1)    
    
    if index == 0:
        array = np.empty(shape=(len(files), data.shape[0], 1))
    array[index] = data

np.save("data/pulseimpute_data/waveforms/eegfmri_vu_ppg/eegfmri_vu_ppg_test.npy", array)
