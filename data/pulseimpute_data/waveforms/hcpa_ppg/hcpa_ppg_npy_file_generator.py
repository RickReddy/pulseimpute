import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import numpy as np
import glob

original_fs = 400 # sample rate is 400
target_fs = 100 # new sample rate
nyquist_freq = target_fs / 2
order = 2
b, a = signal.butter(order, nyquist_freq, fs=original_fs, btype='lowpass')

filepath = "/data1/neurdylab/datasets/hcp_a/physio_preproc/{filename}_preproc/{filename}_preprocVars.mat"
files = []
with open("data/pulseimpute_data/waveforms/hcpa_ppg/hcpa_ppg_good_files.txt") as text_file:
    for line in text_file:
        files.append(filepath.replace("{filename}", line.strip()))

for index, file in enumerate(files):
    mat = sio.loadmat(file)
    data = mat['respRawt']

    data = data.flatten()

    mask = np.ones(data.shape, dtype = bool)
    mask[[0, data.shape[0] - 1]] = 0
    data = np.where(mask & (data == 0), np.nan, data) # hcpa remove 1pt spikes
    # if first or last are 0, equates to NaN leading to errors later on, so exclude those
    
    df = pd.DataFrame(data)
    df = df.interpolate()
    data = df.to_numpy().flatten()
    
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

np.save('data/pulseimpute_data/waveforms/hcpa_ppg/hcpa_ppg_test.npy', array)
