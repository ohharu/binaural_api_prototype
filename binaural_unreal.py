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
    print(len(elev0Hrtf_L[0]))
    for i in range(len(elev0Hrtf_L)):
        """
        hammingWindow = np.hamming(len(elev0Hrtf_L[i]))
        elev0Hrtf_L[i] = elev0Hrtf_L[i] * hammingWindow
        elev0Hrtf_R[i] = elev0Hrtf_R[i] * hammingWindow
        """
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
    
    
    
def hrtfarray(position, hrtf):
    """
    hrtf_array = []
    for i in range(len(hrtf[0])):
        hrtf_array.append(hrtf[position[i]][i])
    """
    hrtf_array = [hrtf[position][i] for i in range(len(hrtf[0]))]
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

def degreecalc(x,y,litenerdeg):
	y = -y
	degree = math.degrees(math.atan2(y, x))
	if degree < 0:
		degree += 360
	degree -= 90
	degree -= litenerdeg
	if degree < 0:
		degree += 360
	print(degree)
	return Map(degree)
	
def Map(value):
    return int(0 + (71 - 0) * ((value - 0) / (360 - 0)))

def play_elev0(wfave, hrtfL, hrtfR, xpos, ypos, deg):
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
        position = degreecalc(xpos,ypos,deg)
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

    