# coding: utf-8

# import Python modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
  
# Read the wavefile: sound of plucking a guitar string
fs,x = wavfile.read('gtr55.wav')

# For Windows, you can use winsound to play the audio file

import winsound
winsound.PlaySound('gtr55.wav', winsound.SND_FILENAME) 
print(type(x))
print(x.size)
print((x.size/fs))

# Time-domain visualization of the signal

t = np.linspace(0,(x.size/fs),x.size)
# plotting
plt.subplot(3, 1, 1)
plt.plot( t, x )
plt.title('Time domain visualization of the audio signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.grid('on')

# Frequency domain visualization of the signal, logarithmic magnitude

max_freq = fs / 2
X = fft(x)
winlen = 1024
frq_scale = np.linspace(0, max_freq, winlen/2.0-1)
mag_scale = 20.0*np.log10(np.abs(X[0:int(winlen/2)-1]))
# plotting
plt.subplot(3, 1, 2)
plt.plot( frq_scale, mag_scale)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')


# Your code here: Spectrogram of the audio signal
f,t,X = scipy.signal.spectrogram(x,fs)
mag_scale = 20*np.log10(1e-6+np.abs(X))
# plotting
plt.subplot(3, 1, 3)
plt.pcolormesh(t,f, mag_scale)
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Log-magnitude spectrogram')
plt.colorbar()

# Show the figure.
plt.tight_layout()
plt.show()



