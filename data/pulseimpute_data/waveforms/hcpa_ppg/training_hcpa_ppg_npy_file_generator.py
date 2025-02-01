import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import numpy as np
import glob

# this file is used to generate data with an 80/10/10 split for retraining the model
# it uses all files available on ACCRE for the HCPA dataset

original_fs = 400 # sample rate is 400
target_fs = 100 # new sample rate
nyquist_freq = target_fs / 2
order = 2
b, a = signal.butter(order, nyquist_freq, fs=original_fs, btype='lowpass')

filepaths = glob.glob('/data1/neurdylab/datasets/hcp_a/physio_preproc/*_preproc/*.mat')
#print(filepaths)

for index, filepath in enumerate(filepaths):
    
    mat = sio.loadmat(filepath)
    data = mat['pulsRawt']
    # if index == 0:
    #     raw_array = np.empty(shape=(len(filepaths), data.shape[0], 1))
    # raw_array[index] = data

    data = data.flatten()
    
    mask = np.ones(data.shape, dtype = bool)
    mask[[0, data.shape[0] - 1]] = 0
    data = np.where(mask & (data == 0), np.nan, data) # hcpa remove 1pt spikes
    # if first or last are 0, equates to NaN leading to errors later on, so exclude those
    
    df = pd.DataFrame(data)
    df = df.interpolate()
    data = df.to_numpy().flatten()

    data = signal.filtfilt(b, a, data) # nyquist lowpass filter

    # data = data.reshape(data.shape[0],1)
    # if index == 0:
    #     filtered_array = np.empty(shape=(len(filepaths), data.shape[0], 1))
    # filtered_array[index] = data
    # data = data.flatten()
    
    time_stamps = pd.date_range(start = 0, freq = f"{round(10**9/original_fs)}N", periods = data.shape[0])
    time_series = pd.Series(data, index = time_stamps)
    time_series = time_series.resample(f"{round(10**9/target_fs)}N").interpolate() # resample to target frequency
    
    data = time_series.to_numpy()
    
    data = data[0:30000]
    # pulseimpute provided missingness patterns are of length 30,000. Cutting off end samples to match

    data = data.reshape(data.shape[0],1)    
    
    if index == 0:
        train_array = np.empty(shape=(round(0.8*len(filepaths)), data.shape[0], 1))
    
    if index < round(0.8*len(filepaths)):
        train_array[index] = data

    if index == round(0.8*len(filepaths)):
        val_array = np.empty(shape=(round(0.1*len(filepaths)), data.shape[0], 1))

    if index >= round(0.8*len(filepaths)) and index < round(0.9*len(filepaths)):    
        val_array[index - round(0.8*len(filepaths))] = data

    if index == round(0.9*len(filepaths)):
        test_array = np.empty(shape=(round(0.1*len(filepaths)), data.shape[0], 1))
    
    if index >= round(0.9*len(filepaths)):
        test_array[index - round(0.9*len(filepaths))] = data

np.save('data/pulseimpute_data/waveforms/hcpa_ppg/hcpa_ppg_train.npy', train_array)
np.save('data/pulseimpute_data/waveforms/hcpa_ppg/hcpa_ppg_val.npy', test_array)
np.save('data/pulseimpute_data/waveforms/hcpa_ppg/hcpa_ppg_test_unsorted.npy', val_array)
