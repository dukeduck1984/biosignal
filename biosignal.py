
# coding: utf-8

import pandas as pd
import numpy as np
from numba import jit
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import os


@jit
def remove_offset(emg_data):
    """
    原始肌电信号均值归零
    Correct the mean of the EMG signal to zero.

    Param:
        emg_data: emg data; type: ndarray
    Return:
        offset_removed: centered EMG data; type: ndarray
    """
    offset_removed = emg_data - np.mean(emg_data)
    return offset_removed


@jit
def rectify(emg_data):
    """
    全波整流
    Full wave rectification of EMG signals.

    Param:
        emg_data: EMG data; type: ndarray
    Return:
        rectified emg signal; type:ndarray
    """
    rectified = np.abs(emg_data)
    return rectified


@jit
def filter_bandpass(signal_data, freq, order=4, highpass=20, lowpass=500):
    """
    带通滤波
    Apply bandpass filter to an EMG or FP signal.

    Param:
        signal_data: EMG or Force Plate data; type: ndarray
        freq: sampling rate (Hz); type: int
        highpass: high pass cut-off (Hz); type: int
        lowpass: low pass cut-off (Hz); type: int
    Return:
        data_filt: filtered EMG or FP data; type: ndarray
    """
    freq_nyq = freq / 2
    high, low = highpass / freq_nyq, lowpass / freq_nyq
    b, a = signal.butter(order, [high, low], btype='bandpass')
    data_filt = signal.filtfilt(b, a, signal_data)
    return data_filt


@jit
def interp_signal(signal_data, original_freq, interp_freq):
    """
    信号采样率插值
    Sampling Rate Interpolate

    Param:
        signal_data: EMG or Force Plate data; type: ndarray
        original_freq: The original sampling rate of the input data; type: int
        interp_freq: The result sampling rate after interpolation; type: int

    Return: new interpolated signal data; type: ndarray
    """
    x = np.arange(signal_data.size)
    y = signal_data[x]
    interp_x = np.linspace(0, signal_data.size, signal_data.size * (interp_freq/original_freq))
    interp_y = np.interp(interp_x, x, y)
    return interp_y


@jit
def filter_lowpass(signal_data, freq, lowpass=3):
    """
    低通滤波
    Apply low pass filter to an EMG or FP signal.

    Param:
        signal_data: EMG or Force Plate data; type: ndarray
        freq: sampling rate (Hz); type: int
        lowpass: low pass cut-off; type: int
    Return:
        low_filt: Filtered EMG or FP data; type: ndarray
    """
    freq_nyq = freq / 2
    high = lowpass / freq_nyq
    b, a = signal.butter(4, high, btype='low')
    low_filt = signal.filtfilt(b, a, signal_data)
    return low_filt


@jit
def calc_rms(emg_data, freq, window_size, overlap='default'):
    """
    计算均方根振幅
    Process an EMG signal using a moving root-mean-square window.

    Param:
        emg_data: emg data; type: ndarray
        freq: sampling rate (Hz); type: int
        window_size: window width (ms); type: int
        overlap: time of overlapping (ms),
                 by default == 1/2 window_size; type: int
    Return: A tuple of:
        rms_final_t: time series of rms; type: ndarray
        rms_final: rms data; type: ndarray
    """
    # By default, set overlap to 1/2 window size
    if overlap == 'default':
        overlap = int(window_size * 0.5)

    sample_width = round(window_size / 1000 * freq)
    emg_data_squared = np.power(emg_data, 2)
    window = np.ones(sample_width) / float(sample_width)
    # Get moving RMS with full superimposing (point by point moving)
    rms_raw = np.sqrt(np.convolve(emg_data_squared, window, 'same'))
    # Number of points per RMS block
    point_size = window_size / 1000 * freq
    midpoint = round(point_size / 2)
    overlap_size = overlap / 1000 * freq
    # Number of RMS blocks out of raw RMS data
    rms_size = 1 + int((rms_raw.size - point_size) / (point_size - overlap_size))

    # Get final RMS with proper overlapping
    rms_final = np.zeros(rms_size)
    for i in range(rms_size):
        rms_final[i] = rms_raw[int(midpoint + i * (point_size - overlap_size))]

    # Get final RMS corresponding time in ms
    rms_final_t = np.zeros(rms_size)
    for t in range(rms_size):
        rms_final_t[t] = round(window_size / 2) + t * (window_size - overlap)

    # Return RMS time series and RMS data in a tuple, both are ndarray
    return (rms_final_t, rms_final)


@jit
def calc_mvc(mvc_data, freq, window_size):
    """
    计算MVC
    Calculate mean MVC value from the highest signal portion (eg. 500ms).

    Param:
        mvc_data: mvc test emg data; type: ndarray
        freq: sampling rate (Hz); type: int
        window_size: window of time (ms); type: int
    Return:
        mvc: MVC amplitude; type: float
    """
    peak_mvc = np.max(mvc_data)
    mvc_index = int(np.where(mvc_data == peak_mvc)[0][0])
    halfwidth = round((window_size / 1000) / 2 * freq)
    mvc = np.mean(mvc_data[mvc_index - halfwidth: mvc_index + halfwidth])
    return mvc


@jit
def onoff_threshold(rms_data, percentage=0.05):
    """
    计算肌电信号的On & Off阈值
    Calculate the Onset & Offset threshold value of the EMG signal.

    Param:
        rms_data: RMS data; type: ndarray
        percentage: percentage of the peak RMS value, eg. typically 0.05 (5%); type: float
    Return:
        threshold: onset & offset amplitude; type: float
    """
    peak_value = np.max(rms_data)
    threshold = peak_value * percentage
    return threshold


@jit
def power_spectrum(emg_data):
    """
    计算功率谱（通过FFT）
    Calculation of the frequency contents - the Total Power Spectrum by FFT.

    Param:
        emg_data: emg data; type: ndarray
    Return: a tuple of:
        power_freq: frequency array of the power spectrum; type: ndarray
        power_spectrum: intensity of the the power spectrum; type: ndarray
        mean_freq: mean frequency of the power spectrum; type: float
        median_freq: median frequency of the power spectrum; type: float
    """
    power_spect = abs(np.fft.rfft(emg_data))
    fft_freq = np.fft.rfftfreq(emg_data.size) * 1000
    power_freq = np.linspace(fft_freq[0], fft_freq[-1], num=fft_freq.size)
    mean_freq = np.sum(power_spect * power_freq)/np.sum(power_spect)
    cumulative_power = []
    for one_power, one_freq in zip(power_spect, power_freq):
        cumulative_power.append(one_power)
        if np.sum(cumulative_power) >= np.sum(power_spect)/2:
            median_freq = one_freq
            break
    return (power_freq, power_spect, mean_freq, median_freq)


@jit
def breath_smooth(breath_data, time_series, window_size=30, method='time'):
    """
    气体代谢数据进行平滑处理
    Smoothing the Gas Exchange data by different methods

    Reference:
    Recommendations for Improved Data Processing from Expired Gas Analysis Indirect Calorimetry
    Robert A et al., 2010
    Sports Medicine

    Param:
        breath_data: Gas data from K4b2; type: ndarray
        time_series: Corresponding time series in seconds (can use 'breath_time_convert' function); type: ndarray
        window_size: time to average in seconds, or number of points (breath). ignored if by lowpass; type: int
        method: 'time': time averaged (in seconds);
                'points': breath running average;
                'lowpass'; 3rd order Butterworth 0.04 lowpass filter;
                type: string
    Return: A tuple of:
        smoothed_time_series: corresponding time series in seconds; type: ndarray
        smoothed_breath_data: smoothed gas exchange data; type: ndarray
     """

    if method == 'lowpass':
        n = 3  # 3rd order Butterworth
        wn = 0.04  # Cutoff frequency
        b, a = signal.butter(n, wn)
        filtered_breath_data = signal.filtfilt(b, a, breath_data)
        # Return breath time series and smoothed breath data in a tuple, both are ndarray
        return (time_series, filtered_breath_data)
    elif method == 'time':
        midpoint = round(window_size / 2)  # Get midpoint of the window, eg. the 8th second if the window is 15s
        smoothed_block = int(time_series[-1] / window_size)  # Get the smoothed block number: total time divided by window size
        smoothed_breath_data = np.zeros(smoothed_block)  # Create ndarray of smoothed data
        smoothed_time_series = np.zeros(smoothed_block)  # Create ndarray of smoothed time series

        for i in range(smoothed_block):
            data_list = []
            for time, data in zip(time_series, breath_data):
                if time > (i * window_size) and time <= (window_size + i * window_size):
                    data_list.append(data)
            data_mean = np.mean(data_list)
            smoothed_breath_data[i] = data_mean

        for n in range(smoothed_block):
            smoothed_time_series[n] = midpoint + n * window_size

        return (smoothed_time_series, smoothed_breath_data)
    elif method == 'points':
        midpoint = round(window_size / 2) # Get midpoint of the window, eg. the 8th breath if the window is 15 breathes
        window = np.ones(window_size)/float(window_size)
        smoothed_breath_data = np.convolve(breath_data, window, 'valid')

        first_time_index = int(midpoint - 1)
        final_time_index = int((midpoint - 1) * -1)
        smoothed_time_series = time_series[first_time_index: final_time_index]

        return (smoothed_time_series, smoothed_breath_data)


@jit
def breath_time_convert(raw_time_string):
    """
    将K4b2导出的excel表中的时间列（t）从字符串转换为整数（单位：秒）
    Convert the time (string format) in K4b2 raw file to seconds (int)

    Param:
        raw_time_string: string in ndarray, example: HH:MM:SS
    Return:
        time_series_sec: seconds in int; type: ndarray
    """
    time_series_sec = []
    for t in raw_time_string:
        h, m, s = t.split(':')
        sec = int(h) * 3600 + int(m) * 60 + int(s)
        time_series_sec.append(sec)
    return np.array(time_series_sec)


@jit
def velocity_time_curve(grf, bw, freq):
    """
    通过GRF一次积分求速度-时间曲线
    Calculate velocity over time series by ground reaction force

    Param:
        grf: Ground Reaction Force obtained from force plate; type: ndarray
        bw: Body weight in kilogram; type: float
        freq: Sampling rate of the force plate; type: int
    Return: ndarray (unit: m/s)
    """
    velocity = (grf - bw * 9.8) * (1 / freq) / bw

    velocity_over_time = []
    for i in range(velocity.size):
        velocity_over_time.append(np.sum(velocity[: i]))
    return np.array(velocity_over_time)


@jit
def distance_time_curve(vtc, freq):
    """
    通过VTC积分求位移-时间曲线（GRF的二次积分）
    Calculate displacement over time series by velocity over time

    Param:
        vtc: Velocity Time Curve; type: ndarray
        freq: Sampling rate of the force plate; type: int
    Return: ndarray (unit: m)
    """
    distance = vtc * 1 / freq
    dist_time_curve = []
    for i in range(distance.size):
        dist_time_curve.append(np.sum(distance[: i]))
    return np.array(dist_time_curve)
