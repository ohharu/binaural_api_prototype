import binaural_unreal
import wave
import pyaudio
import numpy as np
import scipy.fftpack
#import matplotlib.pyplot as plt
import math
#from getch import getch
#import timeout_decorator
import time

filename1 = "Call_Me_Maybe_Carly_Rae_Jepsen_LeadVox.wav"

fs,wf1,wf1ave = binaural_unreal.loadwav(filename1)


hrtf_L, hrtf_R = binaural_unreal.load_elev0hrtf()

xposition = -1 #input relative xpos
yposition = 0 #input relative ypos
listenerdeg = 0 #input listener rotation[degree]

p = pyaudio.PyAudio()

stream = p.open(format = 8,
                channels = 2,
                rate = fs,
                output = True)
index = 0
L = 512
N = 1024
while(wf1[index:].size > L):

    ft = time.time()
    resultData = binaural_unreal.play_elev0(wf1[index:index + N + L], hrtf_L, hrtf_R,xposition,yposition,listenerdeg)
    print(time.time()-ft)
    stream.write(bytes(resultData))
    
    #print(wfave[index:].size,L)
    index += N








