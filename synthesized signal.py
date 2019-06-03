# coding: utf-8

# import Python modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
import winsound 


# Load audio signal
fs, x = wavfile.read('glockenspiel-a-2.wav')
fs = np.float(fs)
number_of_samples = x.size
duration = number_of_samples/fs
print('{} {}'.format('Sampling frequency (in Hz) is: ', fs))
print('{} {}'.format('Duration (in s) is: ', duration))

t_scale = np.linspace(0,(x.size/fs),x.size) 
amp_scale = x

# Plot time domain visualization
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t_scale, amp_scale)
plt.title('Time domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')

# Plot frequency domain visualization using FFT
max_freq = fs/2
win_len = 2048
freq_step = int(win_len / 2) - 1
t_start = 10000
X = fft(x[10000:10000+win_len]);
frq_scale = np.linspace(0, max_freq, win_len/2.0-1)
mag_scale = 20*np.log10(np.abs(X[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

## 2
# Edit the script to generate a Sine wave signal y0,
# with amplitude 3000 (so you will hear something if you play the signal),
# frequency F0, and length of t seconds, as calculated from step 1.

# Generate sine wave y0
F0 = 1877 # To be checked from the previous plot
amplitude = 3000
y0 = amplitude* np.sin(2 * np.pi * F0 * t_scale)

# Plot time domain visualization
short_length = 100
t_scale_short = t_scale[0:short_length]
amp_scale_short = y0[0:short_length]

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(t_scale_short, amp_scale_short)
plt.title('Time domain visualization of the sine wave y0')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (samples)')

# Apply fast Fourier transform (FFT) to the Sine wave. Display the FFT of the waveform.

# Plot frequency domain visualization using FFT
Y0 = fft(y0[10000:10000+win_len])
mag_scale = 20*np.log10(np.abs(Y0[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the sine wave y0')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()


F1 = 7012
amplitude = 1500
y1 =amplitude * np.sin(2 * np.pi * F1 * t_scale)

# Plot time domain visualization 
short_length = 100
t_scale_short = t_scale[0:short_length]
amp_scale_short = y1[0:short_length]

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(t_scale_short, amp_scale_short)
plt.title('Time domain visualization of the sine wave y1')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (samples)')



# Plot frequency domain visualization using FFT
Y1 = fft(y1[10000:10000+win_len])
mag_scale = 20*np.log10(np.abs(Y1[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the sine wave y1')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()


y = y0 + y1
# Plot time domain visualization 
short_length = 100
t_scale_short = t_scale[0:short_length]
amp_scale_short = y[0:short_length]

plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(t_scale_short, amp_scale_short)
plt.title('Time domain visualization of the sine wave y1')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (samples)')



# Plot frequency domain visualization using FFT
Y2 = fft(y[10000:10000+win_len])
mag_scale = 20*np.log10(np.abs(Y2[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the sine wave y1')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()



# To record:
scipy.io.wavfile.write('y0.wav', int(fs), np.int16(y0))
scipy.io.wavfile.write('y1.wav', int(fs), np.int16(y1))
scipy.io.wavfile.write('y.wav', int(fs),np.int16(y))

# To play (on window)
winsound.PlaySound('y0.wav', winsound.SND_FILENAME)
winsound.PlaySound('y1.wav', winsound.SND_FILENAME)
winsound.PlaySound('y.wav',winsound.SND_FILENAME)
