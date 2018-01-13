# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import signal
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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
    interp_x = np.linspace(0, signal_data.size-1, int(signal_data.size * (interp_freq/original_freq)))
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
def breath_smooth(breath_data, time_series, window_size=30, avg_by='time'):
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
        avg_by: 'time': time averaged (in seconds);
                'points': breath running average;
                'lowpass'; 3rd order Butterworth 0.04 lowpass filter;
                type: string
    Return: A tuple of:
        smoothed_time_series: corresponding time series in seconds; type: ndarray
        smoothed_breath_data: smoothed gas exchange data; type: ndarray
     """

    if avg_by == 'lowpass':
        n = 3  # 3rd order Butterworth
        wn = 0.04  # Cutoff frequency
        b, a = signal.butter(n, wn)
        filtered_breath_data = signal.filtfilt(b, a, breath_data)
        # Return breath time series and smoothed breath data in a tuple, both are ndarray
        return (time_series, filtered_breath_data)
    elif avg_by == 'time':
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


@jit
def find_turnpoints(x_data, y_data, initial_g_size=2, step_size=1, turnpoint_size=1, plot=False,
                    tp_initial=0):
    """
    使用线性回归寻找数据的拐点
    Find Turnpoints of Data Points by Brutal-Force Linear Regression

    Param:
        x_data: data on x axis; type: list, or ndarray
        y_data: data on y axis; type: list, or ndarray
        initial_g_size: the initial number of data points in group 1; type: int
        step_size: the increment of group 1; type: int
        turnpoint_size: number of turnpoints expected; type: int
        plot: whether or not plot the data and the regression line; type: boolean
        turnpoints: for data recursion, leave it as is; type: list
        tp_initial: for data recursion, leave it as is; type: int
    Return:
        turnpoints: the global index(es) of data turnpoint(s) after linear regression; type: list
    """
    initialize_tp = []  # Initialize an empty list to collect turnpoints

    def wrapper(x_data, y_data, initial_g_size, step_size, turnpoint_size, plot, tp_initial, turnpoints=initialize_tp):

        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data).reshape(-1, 1)

        residual_sum_of_squares = []  # 收集各分组情况的均方误差列表
        g1_error_list = []  # 第1组数据的误差列表
        g2_error_list = []  # 第2组数据的误差列表
        g1_size = initial_g_size  # 第1组的数据点数
        sample_size = x_data.size  # 总共的数据点数

        while (sample_size - g1_size) >= initial_g_size:  # 当第2组的数据点数大于等于第1组的初始数据点数时
            regr_g1 = linear_model.LinearRegression()
            regr_g1.fit(x_data[:g1_size], y_data[:g1_size])
            # a1, b1 = regr_g1.coef_, regr_g1.intercept_
            g1_mean_sq_error = np.mean((regr_g1.predict(x_data[:g1_size]) - y_data[:g1_size]) ** 2)
            # print('y = {}x + {}'.format(a1[0], b1))
            # plt.plot(log_power[:7], regr_g1.predict(log_power[:7]), color='red')

            regr_g2 = linear_model.LinearRegression()
            regr_g2.fit(x_data[g1_size - 1:], y_data[g1_size - 1:])
            # a2, b2 = regr_g2.coef_, regr_g2.intercept_
            g2_mean_sq_error = np.mean((regr_g2.predict(x_data[g1_size - 1:]) - y_data[g1_size - 1:]) ** 2)
            # print('y = {}x + {}'.format(a2[0], b2))
            # plt.plot(log_power[4:], regr_g2.predict(log_power[4:]), color='green')

            residual_sum_of_squares.append(g1_mean_sq_error + g2_mean_sq_error)  # 两组的均方误差之和加入列表
            g1_error_list.append(g1_mean_sq_error)
            g2_error_list.append(g2_mean_sq_error)

            g1_size += step_size

        # print(residual_sum_of_squares)
        # 通过最小均方误差找到拐点index
        # rss = np.array(residual_sum_of_squares)
        # turnpoint_index = initial_g_size - 1 + step_size * np.where(rss == np.min(rss))[0]
        min_error_index = residual_sum_of_squares.index(min(residual_sum_of_squares))

        g1_min_error = g1_error_list[min_error_index]
        g2_min_error = g2_error_list[min_error_index]

        # 获得分组内的拐点index
        turnpoint_index = initial_g_size - 1 + step_size * min_error_index

        # 获得全局的拐点index
        if g1_min_error < g2_min_error:
            tp_global = initial_g_size - 1 + step_size * min_error_index + tp_initial
            tp_initial = turnpoint_index + tp_initial
        else:
            tp_global = initial_g_size - 1 + step_size * min_error_index + tp_initial

        # 把全局拐点index加入列表，同时避免重复记录
        if tp_global not in turnpoints:
            turnpoints.append(tp_global)
        tp2 = turnpoints

        # 把确认找到的拐点再做一次拟合，用于输出图像
        regr_g1 = linear_model.LinearRegression()
        regr_g1.fit(x_data[:turnpoint_index + 1], y_data[:turnpoint_index + 1])
        # a1, b1 = regr_g1.coef_, regr_g1.intercept_
        # print('y = {}x + {}'.format(a1[0], b1))

        regr_g2 = linear_model.LinearRegression()
        regr_g2.fit(x_data[turnpoint_index:], y_data[turnpoint_index:])
        # a2, b2 = regr_g2.coef_, regr_g2.intercept_
        # print('y = {}x + {}'.format(a2[0], b2))

        if plot:  # 是否要作图的选项
            if turnpoint_size > 1:
                if g1_min_error > g2_min_error:
                    # plt.plot(x_data[:turnpoint_index+1+step_size], y_data[:turnpoint_index+1+step_size], 'o')
                    plt.plot(x_data[turnpoint_index - step_size:], y_data[turnpoint_index - step_size:], 'o')
                    # plt.plot(x_data[:turnpoint_index+1+step_size], regr_g1.predict(x_data[:turnpoint_index+1+step_size]))
                    plt.plot(x_data[turnpoint_index - step_size:],
                             regr_g2.predict(x_data[turnpoint_index - step_size:]))
                else:
                    plt.plot(x_data[:turnpoint_index + 1 + step_size], y_data[:turnpoint_index + 1 + step_size], 'o')
                    # plt.plot(x_data[turnpoint_index-step_size:], y_data[turnpoint_index-step_size:], 'o')
                    plt.plot(x_data[:turnpoint_index + 1 + step_size],
                             regr_g1.predict(x_data[:turnpoint_index + 1 + step_size]))
                    # plt.plot(x_data[turnpoint_index-step_size:], regr_g2.predict(x_data[turnpoint_index-step_size:]))
            else:
                plt.plot(x_data[:turnpoint_index + 1 + step_size], y_data[:turnpoint_index + 1 + step_size], 'o')
                plt.plot(x_data[turnpoint_index - step_size:], y_data[turnpoint_index - step_size:], 'o')
                plt.plot(x_data[:turnpoint_index + 1 + step_size],
                         regr_g1.predict(x_data[:turnpoint_index + 1 + step_size]))
                plt.plot(x_data[turnpoint_index - step_size:], regr_g2.predict(x_data[turnpoint_index - step_size:]))

        # 下面的语句把1个或者2个拐点的index存入turnpoints列表，并返回
        while turnpoint_size > 1:
            if g1_min_error > g2_min_error:
                x_data = x_data[:turnpoint_index + 1 + step_size]
                y_data = y_data[:turnpoint_index + 1 + step_size]
                wrapper(x_data, y_data, initial_g_size=initial_g_size, step_size=step_size,
                        turnpoint_size=turnpoint_size - 1, plot=plot, turnpoints=tp2, tp_initial=tp_initial)
                return turnpoints
                turnpoint_size -= 1
            else:
                x_data = x_data[turnpoint_index + step_size:]
                y_data = y_data[turnpoint_index + step_size:]
                wrapper(x_data, y_data, initial_g_size=initial_g_size, step_size=step_size,
                        turnpoint_size=turnpoint_size - 1, plot=plot, turnpoints=tp2, tp_initial=tp_initial)
                return turnpoints
                turnpoint_size -= 1
        else:
            return turnpoints

    result = wrapper(x_data=x_data, y_data=y_data, initial_g_size=initial_g_size, step_size=step_size,
                     turnpoint_size=turnpoint_size, plot=plot, turnpoints=[], tp_initial=tp_initial)

    return result


@jit
def find_lt_loglog_method(intensity_data, lactate_data, plot=False):
    """
    使用Log-log方法判定乳酸阈（LT1）

    Reference:
        Beaver, W. L., Wasserman, K. A. R. L. M. A. N., & Whipp, B. J. (1985).
        Improved detection of lactate threshold during exercise using a log-log transformation.
        Journal of applied physiology, 59(6), 1936-1940.
    Param:
        intensity_data: data series of test load (power, or speed); type: list, or ndarray
        lactate_data: data series of BLa; type: list, or ndarray
        plot: whether or not plot the data and the regression line; type: boolean
    Return:
        (Load at LT1, BLa at LT1, Index of data at LT1); type: tuple
    """
    intensity_data = np.array(intensity_data).reshape(-1, 1)
    lactate_data = np.array(lactate_data).reshape(-1, 1)
    log_intensity = np.log(intensity_data)  # 计算运动强度数据的log
    log_lactate = np.log(lactate_data)  # 计算血乳酸数据的log
    # 找到LT1拐点的index及两条拟合直线的斜率和截距
    lt1_index = find_turnpoints(log_intensity, log_lactate, plot=plot)
    regr_g1 = linear_model.LinearRegression()
    regr_g1.fit(log_intensity[:lt1_index[0] + 1], log_lactate[:lt1_index[0] + 1])
    a1, b1 = regr_g1.coef_, regr_g1.intercept_
    # print('y = {}x + {}'.format(a1[0], b1))
    regr_g2 = linear_model.LinearRegression()
    regr_g2.fit(log_intensity[lt1_index[0]:], log_lactate[lt1_index[0]:])
    a2, b2 = regr_g2.coef_, regr_g2.intercept_
    a1 = a1[0][0]
    a2 = a2[0][0]
    b1 = b1[0]
    b2 = b2[0]
    x = (b2 - b1) / (a1 - a2)
    y = a1 * (b2 - b1) / (a1 - a2) + b1
    lt1_intensity = np.exp(x)  # 计算两条拟合直线交点对应的X轴数据，即LT1运动强度
    lt1_lactate = np.exp(y)  # 计算两条拟合直线交点对应的Y轴数据，即LT1血乳酸值
    return (lt1_intensity, lt1_lactate, lt1_index[0])  # 返回元组（LT1的运动强度, LT1的血乳酸值, LT1的index）


@jit
def find_lt_sds_method(intensity_data, lactate_data, loglog=False, plot=False):
    """
    使用标准化Dmax确定LT1 & LT2

    Reference:
        Standardization of the Dmax Method for Calculating the Second Lactate Threshold
        Chalmers et al., 2015
        International Journal of Sports Physiology and Performance
    Param:
        intensity_data: data series of test load (power, or speed); type: list, or ndarray
        lactate_data: data series of BLa; type: list, or ndarray
        log-log: whether or not to use Log-log transformation to determin the LT1, if false, LT1 is determined by first rise of 0.4mM BLa; type: boolean
        plot: whether or not plot the data and the regression line; type: boolean
    Return:
        (Load at LT1, BLa at LT1, Index of data at LT1, Load at LT2, BLa at LT2); type: tuple
    """
    intensity_data = np.array(intensity_data).reshape(-1, 1)
    lactate_data = np.array(lactate_data).reshape(-1, 1)

    if loglog:  # 如果使用loglog transformation计算LT1
        lt1_intensity, lt1_lactate, lt1_index = find_lt_loglog_method(intensity_data, lactate_data, plot=False)
    else:  # 使用La浓度初次超过0.4mM的前一级负荷作为LT1
        for index, la in enumerate(lactate_data):
            if index > 0:
                if (lactate_data[index] - lactate_data[index - 1]) >= 0.4:
                    lt1_index = index - 1
                    break
        lt1_intensity = intensity_data[lt1_index][0]
        lt1_lactate = lactate_data[lt1_index][0]

    # 获得LT1点的X和Y的坐标
    lt1_x = intensity_data[lt1_index][0]
    lt1_y = lactate_data[lt1_index][0]

    # 获得血乳酸最大值那点的X和Y的坐标
    la_final_x = intensity_data[-1][0]
    la_final_y = lactate_data[-1][0]

    # 设LT1点的坐标为（x1, y1）
    # x1 = power[1][0]
    # y1 = lactate[1][0]

    if lt1_index >= 2:
        x1 = intensity_data[lt1_index - 2][0]
        y1 = lactate_data[lt1_index - 2][0]
        curvefit_index = lt1_index - 2
    elif lt1_index >= 1 and lt1_index < 2:
        x1 = intensity_data[lt1_index - 1][0]
        y1 = lactate_data[lt1_index - 1][0]
        curvefit_index = lt1_index - 1
    else:
        # print(power[lt1_index], lactate[lt1_index])
        x1 = intensity_data[lt1_index][0]
        y1 = lactate_data[lt1_index][0]
        curvefit_index = lt1_index

    # x1 = lt1_x
    # y1 = lt1_y

    # 设血乳酸最大值的坐标为（x2, y2）
    x2 = la_final_x
    y2 = la_final_y

    # 根据两点式推导出一般式 AX+BY+C=0的A, B, C
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    # 多项式回归线上一点到直线的距离
    # d = np.abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)

    # 曲线拟合的X和Y数据序列
    x_train = intensity_data[curvefit_index:]
    y_train = lactate_data[curvefit_index:]
    la_curve_x = np.linspace(intensity_data[curvefit_index][0], x2, 300)
    d_mod_line_x = np.linspace(x1, x2, 300)

    poly = PolynomialFeatures(degree=3)  # 3次多项式回归
    x_train_quadratic = poly.fit_transform(x_train)
    regressor_quadratic = linear_model.LinearRegression()
    regressor_quadratic.fit(x_train_quadratic, y_train)
    d_mod_line = linear_model.LinearRegression()
    d_mod_line.fit([[x1], [x2]], [[y1], [y2]])
    xx_quadratic = poly.transform(la_curve_x.reshape(la_curve_x.shape[0], 1))
    la_curve_x = la_curve_x.reshape(-1, 1)
    d_mod_line_x = d_mod_line_x.reshape(-1, 1)

    # 建一个列表来储存d值
    find_d_max_list = []

    # 遍历弧线上的点
    for x0, y0 in zip(la_curve_x, regressor_quadratic.predict(xx_quadratic)):
        x0 = x0[0]
        y0 = y0[0]
        # 求弧线上点（x0, y0）到直线的距离 d
        d = np.abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
        # 如果列表为空，则储存（x0, y0）坐标和 d 值
        if len(find_d_max_list) == 0:
            find_d_max_list.append(x0)  # X0坐标，即运动强度
            find_d_max_list.append(y0)  # Y0坐标，即血乳酸浓度
            find_d_max_list.append(d)  # d，即Dmax值
        # 如列表已有数据，则比较 d 的大小并进行替换，直至找到 d 的最大值
        elif len(find_d_max_list) == 3:
            if d > find_d_max_list[-1]:
                find_d_max_list[0] = x0
                find_d_max_list[1] = y0
                find_d_max_list[2] = d

    lt2_intensity = find_d_max_list[0]
    lt2_lactate = find_d_max_list[1]

    if plot:
        plt.plot(intensity_data, lactate_data, '^')  # 画出负荷强度vs血乳酸的原始数据点
        plt.plot(la_curve_x, regressor_quadratic.predict(xx_quadratic), 'r-')  # 画出血乳酸呈指数上升的回归曲线
        plt.plot(d_mod_line_x, d_mod_line.predict(d_mod_line_x), 'g--')  # 画出连接第一个和最后一个数据点的直线
        plt.plot([lt1_intensity, lt1_intensity], [0, lt1_lactate], '--')  # 画出LT1的标示线
        plt.plot([lt2_intensity, lt2_intensity], [0, lt2_lactate], '--')  # 画出LT2的标示线

    # 返回元组（LT1的运动强度, LT1的血乳酸值, LT1的index, LT2的运动强度, LT2的血乳酸值）
    return (lt1_intensity, lt1_lactate, lt1_index, lt2_intensity, lt2_lactate)


@jit
def find_lt_dickhuth_method(intensity_data, lactate_data, plot=False):
    """
    使用Dickhuth-Berg方法确定LT1 & LT2

    Reference:
        Berg A, Jokob M, Lehmann HH, Dickhuth G, Huber J. Actualle Aspekte der modernen ergometrie. Pneum 1990;44:2-13.
    Param:
        intensity_data: data series of test load (power, or speed); type: list, or ndarray
        lactate_data: data series of BLa; type: list, or ndarray
        plot: whether or not plot the data and the regression line; type: boolean
    Return:
        (Load at LT1, BLa at LT1, Index of data at LT1, Load at LT2, BLa at LT2); type: tuple
    """
    intensity_data = np.array(intensity_data).reshape(-1, 1)
    lactate_data = np.array(lactate_data).reshape(-1, 1)

    # 计算乳酸浓度/负荷比
    lactate_eq = (lactate_data / intensity_data).reshape(-1)
    # 找到最小乳酸浓度/负荷比，即LT1的index
    lt1_index = np.argsort(lactate_eq)[0]
    # 获得LT1的乳酸浓度
    lt1_lactate = lactate_data[lt1_index][0]
    lt1_intensity = intensity_data[lt1_index][0]
    # 获得LT2的乳酸浓度，即LT1 + 1.5mM
    lt2_lactate = lt1_lactate + 1.5

    # 根据LT1点的index决定曲线拟合的起点index
    if lt1_index >= 2:
        curvefit_index = lt1_index - 2
    elif lt1_index >= 1 and lt1_index < 2:
        curvefit_index = lt1_index - 1
    else:
        curvefit_index = lt1_index

    # LT1 点的X坐标
    lt1_x = intensity_data[lt1_index][0]
    # 第一级和最后一级的X坐标
    x1 = intensity_data[0][0]
    x2 = intensity_data[-1][0]

    x_train = intensity_data[curvefit_index:]
    y_train = lactate_data[curvefit_index:]
    la_curve_x = np.linspace(intensity_data[curvefit_index][0], x2, 300)

    poly = PolynomialFeatures(degree=3)  # 3次多项式回归
    x_train_quadratic = poly.fit_transform(x_train)
    regressor_quadratic = linear_model.LinearRegression()
    regressor_quadratic.fit(x_train_quadratic, y_train)
    xx_quadratic = poly.transform(la_curve_x.reshape(la_curve_x.shape[0], 1))
    la_curve_x = la_curve_x.reshape(-1, 1)

    # 在拟合的曲线上找到LT2的index和对应的强度
    lt2_index = np.argsort(np.abs(regressor_quadratic.predict(xx_quadratic).reshape(-1) - lt2_lactate))[0]
    lt2_intensity = la_curve_x[lt2_index][0]

    if plot:
        plt.plot(intensity_data, lactate_data, '^')  # 画出负荷强度vs血乳酸的原始数据点
        plt.plot(la_curve_x, regressor_quadratic.predict(xx_quadratic), 'r-')  # 画出血乳酸呈指数上升的回归曲线
        plt.plot([lt1_x, lt1_x], [0, lt1_lactate], '--')  # 画出LT1的标示线
        plt.plot([lt2_intensity, lt2_intensity], [0, lt2_lactate], '--')  # 画出LT2的标示线

        plt.ylabel('Lactate (mM)')

        pace_list = []
        for kph in range(x1, x2 + 1):
            m, s = str(1 / (kph / 60)).split(".")
            s = str(int(float("0." + s) * 60))
            if s == "0":
                s = "00"
            pace_list.append(m + ":" + s)

            # plt.xlabel('Pace (min/km)')
            # plt.xticks(range(x1, x2 + 1), pace_list)

    return (lt1_intensity, lt1_lactate, lt1_index, lt2_intensity, lt2_lactate)


def oxidation_factor(vco2, vo2):
    """
    输入VCO2和VO2得到呼吸商，再得到糖和脂肪的氧化供能占比
    Get RER, and energy source by VO2 and VCO2 data

    Reference: PERONNET F et al., 1991

    Param:
        vco2: vco2 data from gas exchange test; type: ndarray
        vo2: vo2 data from gas exchange test; type: ndarray
    Return: A tuple:
        rer: Respiratory Exchange Ratio; type:float
        glucose_kcal: Energy provided by glucose (kcal/min); type: ndarray
        fat_kcal: Energy provided by fat (kcal/min); type: ndarray
        glucose_percent: Energy provided by glucose percentage; type: ndarray
        fat_percent: Energy provided by fat percentage; type: ndarray
    Example:
        glucose, fat = oxidation_factor(3282.49489346189, 3568.79672239863)
        print("糖氧化为{}%；脂肪氧化为{}%".format(glucose, fat))
    """
    rq = vco2 / vo2 # 得到呼吸商原始值
    rer = np.where(rq >= 0.7, rq, 0.7) # 排除异常值，即将呼吸商最小值设为0.7
    rq = np.where(rer <= 1, rer, 1) # 用于计算有氧部分的供能来源，故将呼吸商最大值设为1
    x_gram_fat = (0.7426 - 0.7455 * rq) / (2.0092 * rq - 1.4136) # x g脂肪氧化产生的能量
    e = 3.8683 + x_gram_fat * 9.7460 # 氧化1g糖和x g脂肪产生的能量 （PERONNET F et al., 1991）
    glucose_percent = 3.8683 / e * 100 # 糖有氧氧化供能百分比
    fat_percent = 100 - glucose_percent # 脂肪有氧氧化供能百分比
    energy_oxidation = e / (0.7455 + 2.0092 * x_gram_fat) # 有氧氧化供能的能量（Kcal/L）
    glucose_kcal = energy_oxidation * vo2 / 1000 * glucose_percent / 100 # 糖有氧氧化供能的能量（Kcal/min）
    fat_kcal = energy_oxidation * vo2 / 1000 * fat_percent / 100 # 脂肪有氧氧化供能的能量（Kcal/min）
    return (rer, glucose_kcal, fat_kcal, glucose_percent, fat_percent) # 返回数组（呼吸商，糖供能能量，脂肪供能能量，糖供能百分比，脂肪供能百分比）
