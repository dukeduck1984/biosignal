# biosignal
# A Python Library for Processing Bio Signal

## 1. Processing EMG Signal

1.1 Remove Offset of EMG Amplitude:
###### The mean amplitude of the EMG signal should be zero, so if the mean has an offset, it should be removed prior to any further analysis.
```
remove_offset(emg_data)

Param:
    emg_data: emg data; type: ndarray
Return:
    offset_removed: centered EMG data; type: ndarray
```
1.2 Rectify the EMG Signal:
###### Also known as "Full Wave Rectify", by getting the absolute value of the EMG signal.
```
rectify(emg_data)

Param:
    emg_data: EMG data; type: ndarray
Return:
    rectified emg signal; type:ndarray
```

1.3 Filtering the EMG signal by 4th order Butterworth Bandpass Filter
###### By default, this bandpass filter uses 500hz for lowpass, and 20hz for highpass filter, the order for Butterworth is 4th.
```
filter_bandpass(signal_data, freq, order=4, highpass=20, lowpass=500)

Param:
    signal_data: EMG or Force Plate data; type: ndarray
    freq: sampling rate (Hz); type: int
    highpass: high pass cut-off (Hz); type: int
    lowpass: low pass cut-off (Hz); type: int
Return:
    data_filt: filtered EMG or FP data; type: ndarray
```
