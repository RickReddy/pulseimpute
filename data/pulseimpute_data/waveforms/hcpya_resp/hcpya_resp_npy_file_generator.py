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

filepath = "/data1/datasets/hcp/{file_id}/MNINonLinear/Results/{file_paradigm}/{file_paradigm}_Physio_log.txt"
files = []
with open("data/pulseimpute_data/waveforms/hcpya_resp/hcpya_resp_good_files.txt") as text_file:
    for line in text_file:
        file_id = line.strip().split("-")[0]
        file_paradigm = line.strip().split('-', 1)[1].replace("-", "_")
        files.append(filepath.replace("{file_id}", file_id).replace("{file_paradigm}", file_paradigm))

for index, file in enumerate(files):
    df = pd.read_csv(file, sep='\s{1,}', engine='python')
    data = df.iloc[:, 1].to_numpy()

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

np.save('data/pulseimpute_data/waveforms/hcpya_resp/hcpya_resp_test.npy', array)
