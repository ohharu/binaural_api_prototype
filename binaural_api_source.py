import wave
import pyaudio
import numpy as np
import scipy.fftpack
#import matplotlib.pyplot as plt
import math
import time

# パラメータ設定
r = 0.03  # マイク間距離
N = 1024  # 点
fs = 44100  # サンプリング周波数
df = int(fs / N)  # サンプリング間隔
c = 340  # 音速
L = 512  # フィルタ長
key = 0  # 首振りの値


def loadwav(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()  # サンプリング周波数
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16")
    # x = np.frombuffer(x, dtype= "int16") / 32768.0  # -1 - +1に正規化
    xwav = 1
    wf.close()
    return fs, x, xwav



# HRTF読み込み
def load_elev0hrtf():
    elev0Hrtf_L = {}
    elev0Hrtf_R = {}

    for i in range(72):
        str_i = str(i * 5)

        if len(str_i) < 2:
            str_i = "00" + str_i
        elif len(str_i) < 3:
            str_i = "0" + str_i

        fileName = "L0e" + str_i + "a.dat"
        filePath = "hrtfs/elev0/" + fileName
        test = open(filePath, "r").read().split("\n")

        data = []

        for item in test:
            if item != '':
                data.append(float(item))

        elev0Hrtf_L[i] = data

    for i in range(72):
        str_i = str(i * 5)

        if len(str_i) < 2:
            str_i = "00" + str_i
        elif len(str_i) < 3:
            str_i = "0" + str_i

        fileName = "R0e" + str_i + "a.dat"
        filePath = "hrtfs/elev0/" + fileName
        test = open(filePath, "r").read().split("\n")

        data = []

        for item in test:
            if item != '':
                data.append(float(item))
        elev0Hrtf_R[i] = data
    # plt.plot(elev0Hrtf_L[0])
    # plt.show()
    hrtf_L_fft = []
    hrtf_R_fft = []

    # HRTFのFFT
    for i in range(len(elev0Hrtf_L)):
        hammingWindow = np.hamming(len(elev0Hrtf_L[i]))
        elev0Hrtf_L[i] = elev0Hrtf_L[i] * hammingWindow
        elev0Hrtf_R[i] = elev0Hrtf_R[i] * hammingWindow
        fl = np.fft.fft(elev0Hrtf_L[i], n=N)
        fr = np.fft.fft(elev0Hrtf_R[i], n=N)
        hrtf_L_fft.append(fl)
        hrtf_R_fft.append(fr)
    """a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in hrtf_L_fft[0]]	
    plt.plot(a)
    plt.show()
    hrtf = np.fft.ifft(hrtf_L_fft[0],n=N).real
    plt.plot(hrtf)
    plt.show()"""

    return hrtf_L_fft, hrtf_R_fft


def calcdegree(r, N, fs, df, c, wf1, wf2, wf3, wf4, key):
    hammingWindow = np.hamming(len(wf1))
    wf1 = wf1 * hammingWindow
    wf2 = wf2 * hammingWindow
    wf3 = wf3 * hammingWindow
    wf4 = wf4 * hammingWindow

    X1 = np.fft.fft(wf1, n=N)
    X2 = np.fft.fft(wf2, n=N)
    X3 = np.fft.fft(wf3, n=N)
    X4 = np.fft.fft(wf4, n=N)

    amplitudeSpectrum1 = np.abs(X1)  # 振幅スペクトル
    phaseSpectrum2 = np.angle(X2)  # 位相スペクトル

    amplitudeSpectrum3 = np.abs(X3)  # 振幅スペクトル
    phaseSpectrum4 = np.angle(X4)  # 位相スペクトル

    crossSpectrum = []
    """
    for i in range(len(phaseSpectrum)):
        value = (phaseSpectrum2[i]-phaseSpectrum[i]) * np.pi / 180
        crossSpectrum.append(value)
    """
    value = (phaseSpectrum4 - phaseSpectrum2) * np.pi / 180
    crossSpectrum = np.append(crossSpectrum, value)

    # 周波数bin毎の到達時間差
    at = []
    at.append(0)
    for i in range(N - 1):
        at.append(crossSpectrum[i + 1] * 1 / (2 * np.pi * df * (i + 1)))  # 周波数bin毎の到達時間差
    # at = crossSpectrum /

    # 到来角度計算
    de = []
    for i in range(N):
        value = c * at[i] / r
        if value > 1 or -1 > value:
            value = 0

        degreevalue = 180 * math.asin(value) / np.pi
        copydegree = degreevalue
        if degreevalue < 0:
            degreevalue = 360 + degreevalue

        if degreevalue >= 0 and 90 >= degreevalue and amplitudeSpectrum3[i] < amplitudeSpectrum1[i]:
            degreevalue += 180 - 2 * abs(copydegree)
        elif degreevalue > 90 and 180 >= degreevalue and amplitudeSpectrum3[i] > amplitudeSpectrum1[i]:
            degreevalue -= 180 - 2 * abs(copydegree)
        elif degreevalue > 180 and 270 >= degreevalue and amplitudeSpectrum3[i] > amplitudeSpectrum1[i]:
            degreevalue += 180 - 2 * abs(copydegree)
        elif degreevalue > 270 and 360 >= degreevalue and amplitudeSpectrum3[i] < amplitudeSpectrum1[i]:
            degreevalue -= 180 - 2 * abs(copydegree)
        de.append(degreevalue)


    degree = [int(val / 5) + key for val in de]

    for i in range(len(de)):
        a = de[i] % 5

        if a > 2.5:
            degree[i] += 1
        while (True):
            if degree[i] >= 72:
                degree[i] = degree[i] - 72
            elif degree[i] < 0:
                degree[i] = degree[i] + 72
            else:
                break

    return degree


def hrtfarray(position, hrtf):
    """
    hrtf_array = []
    for i in range(len(hrtf[0])):
        hrtf_array.append(hrtf[position[i]][i])
    """
    hrtf_array = [hrtf[position[i]][i] for i in range(len(hrtf[0]))]
    return hrtf_array


# hrtfの畳み込み
def convolution(data, hrtf_fft, N, L):
    #ft = time.time()
    tmpFilter = np.zeros(N)
    # tmpFilter[L:] += hrtf
    hammingWindow = np.hamming(len(data))
    data = data * hammingWindow
    spectrum = np.fft.fft(data, n=N)
    add = spectrum * hrtf_fft
    result = np.fft.ifft(add, n=N).real
    # hrtf = np.fft.ifft(hrtf_fft,n=N).real
    # plt.plot(result)
    # plt.show()
    # a = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in hrtf_fft]
    # plt.plot(a)
    # plt.show()

    return result[L:]


def play_elev0(wfave, wf1, wf2, wf3, wf4, hrtfL, hrtfR, key):
    r = 0.03  # マイク間距離
    N = 1024  # 点
    fs = 44100  # サンプリング周波数
    df = int(fs / N)  # サンプリング間隔
    c = 340  # 音速
    L = 512  # フィルタ長
    index = 0
    f = 0
    # resultData = np.empty((0, 2), dtype=np.int16)
    # resultData2 = np.empty((0, 2), dtype=np.int16)
    while (wfave[index:].size > L):
        f += 1
        # ft = time.time()
        if f == 1:
            resultData = np.empty((0, 2), dtype=np.int16)
        position = calcdegree(r, N, fs, df, c, wf1[index:index + N], wf2[index:index + N], wf3[index:index + N],
                                   wf4[index:index + N], key)
        # print(time.time()-ft)
        #ft = time.time()
        virhrtf_L = hrtfarray(position, hrtfL)
        virhrtf_R = hrtfarray(position, hrtfR)

        # ft = time.time()
        convL = convolution(wfave[index:index + N], virhrtf_L, N, L)
        convR = convolution(wfave[index:index + N], virhrtf_R, N, L)
        # print(time.time() - ft)
        # convL = wfave[index:index + N]
        # convR = wfave[index:index + N]
        """
        for i in range(convL.size):
            resultData = np.append(resultData, np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16), axis=0)
        """
        if f == 1:
            resultData = [np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16) for i in range(convL.size)]
        elif f == 2:
            resultData = np.append(resultData,[np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16) for i in range(convL.size)])
            f = 0
            return resultData
       
        # print(len(resultData))
        # print(time.time()-ft)
        # resultData2 = np.append(resultData2,resultData)
        index += L
        # print(f)

def binaural_recording(hrtfL, hrtfR, key,position):
    N = 1024
    L = 512
    while (wfave[index:].size > L):
        f += 1

        virhrtf_L = hrtfarray(position, hrtfL)
        virhrtf_R = hrtfarray(position, hrtfR)

        # ft = time.time()
        convL = convolution(wfave[index:index + N], virhrtf_L, N, L)
        convR = convolution(wfave[index:index + N], virhrtf_R, N, L)
        # print(time.time() - ft)
        # convL = wfave[index:index + N]
        # convR = wfave[index:index + N]
        """
        for i in range(convL.size):
            resultData = np.append(resultData, np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16), axis=0)
        """
        if f == 1:
            resultData = [np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16) for i in range(convL.size)]
        elif f == 2:
            resultData = np.append(resultData,[np.array([[int(convL[i]), int(convR[i])]], dtype=np.int16) for i in range(convL.size)])
            f = 0
            return resultData
       
        # print(len(resultData))
        # print(time.time()-ft)
        # resultData2 = np.append(resultData2,resultData)
        index += L
        # print(f)


#wfave = -wf1 - wf2 + wf3 + wf4



