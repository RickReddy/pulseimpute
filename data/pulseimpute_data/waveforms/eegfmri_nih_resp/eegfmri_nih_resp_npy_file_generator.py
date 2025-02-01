import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import numpy as np
import glob

original_fs = 2000 # original sample rate
target_fs = 100 # new sample rate
nyquist_freq = target_fs / 2
order = 2
b, a = signal.butter(order, nyquist_freq, fs=original_fs, btype='lowpass')

filepath = "/data1/neurdylab/datasets/eegfmri_nih/RAW/{filename}/phys/proc/"
files = []
with open("data/pulseimpute_data/waveforms/eegfmri_nih_resp/eegfmri_nih_resp_good_files.txt") as text_file:
    for line in text_file:
        files = files + glob.glob(filepath.replace("{filename}", line.strip()) + "*.mat")

for index, file in enumerate(files):
    mat = sio.loadmat(file)
    data = mat['OUT_p'][0]['resp'][0]['wave'][0][0]

    data = data.flatten()
    data = signal.filtfilt(b, a, data) # nyquist lowpass filter

    time_stamps = pd.date_range(start = 0, freq = f"{round(10**9/original_fs)}N", periods = data.shape[0])
    time_series = pd.Series(data, index = time_stamps)
    time_series = time_series.resample(f"{round(10**9/target_fs)}N").interpolate() # resample to target frequency
    data = time_series.to_numpy()
    
    data = data[0:30000]
    # pulseimpute provided missingness patterns are of length 30,000. Cutting off end samples to match

    data = data.reshape(data.shape[0],1)    
    
    if index == 0:
        array = np.empty(shape=(len(files), data.shape[0], 1))
    array[index] = data

np.save('data/pulseimpute_data/waveforms/eegfmri_nih_resp/eegfmri_nih_resp_test.npy', array)
