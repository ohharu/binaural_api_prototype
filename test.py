import binaural_api_source
import wave
import pyaudio
import numpy as np
import scipy.fftpack
#import matplotlib.pyplot as plt
import math
#from getch import getch
#import timeout_decorator
import time

filename1 = "1.wav"
filename2 = "2.wav"
filename3 = "3.wav"
filename4 = "4.wav"
fs,wf1,wf1ave = binaural_api_source.loadwav(filename1)
a,wf2,wf2ave = binaural_api_source.loadwav((filename2))
a,wf3,wf3ave = binaural_api_source.loadwav((filename3))
a,wf4,wf4ave = binaural_api_source.loadwav((filename4))
wfave = -wf1 - wf2 + wf3 + wf4

hrtf_L, hrtf_R = binaural_api_source.load_elev0hrtf()
key = 0

p = pyaudio.PyAudio()

stream = p.open(format = 8,
                channels = 2,
                rate = fs,
                output = True)
index = 0
L = 512
N = 1024
while(wfave[index:].size > L):

    ft = time.time()
    resultData = binaural_api_source.play_elev0(wfave[index:index + N + L], wf1[index:index + N + L], wf2[index:index + N + L], wf3[index:index + N + L], wf4[index:index + N + L], hrtf_L, hrtf_R, key)
    print(time.time()-ft)
    stream.write(bytes(resultData))
    
    #print(wfave[index:].size,L)
    index += N








