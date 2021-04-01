# [2021-04-01 Created by Yuzhong WU, The Chinese University of Hong Kong, Hong Kong]
# Feature Extraction for Acoustic Scene Classification task
# 1. logmel
# 2. wavelet based filterbank feature (The so-called "scalogram" in Table 4 in the paper https://ieeexplore.ieee.org/abstract/document/9053194)
# 3. Decomposing the waveletFB feature using temporal median filtering.

import numpy as np
import soundfile as sf
import librosa #use to extract MFCC feature
import scipy
import imageio
from kymatio.scattering1d.filter_bank import scattering_filter_factory # It is used to generate wavelet filters.
import pdb

def save_TF_img(tfimg, filename):
	specgram_img = tfimg.T
	specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
	specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
	specgram_img = specgram_img.astype(np.uint8)
	imageio.imwrite(filename + '.jpg', specgram_img)
	return


#===============
# Configuration
#===============
# Config: STFT
n_fft = 2048
win_length_in_sec = 0.025
hop_length_in_sec = 0.010

# Config: logmel
n_mels = 128

# Config: wavelet filterbank feature
J=10 # The parameter `J` specifies the maximum scale of the filters as a power of two. In other words, the largest filter will be concentrated in a time interval of size `2**J`.
Q=16 # The `Q` parameter controls the number of wavelets per octave in the first-order filter bank.

#===============
# Feature Extraction
#===============

# Load an audio file
data, samplerate = sf.read('airport-barcelona-0-0-a.wav') # 10-second monaural audio.

# Extract the power spectrogram
win_length =  int(win_length_in_sec * samplerate)
hop_length =  int(hop_length_in_sec * samplerate)
window = scipy.signal.hamming(win_length, sym=False)
power_spectrogram = np.abs(librosa.stft(data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=True, window=window))**2

#---------
# 1. Extract the log-mel feature
#---------
# Create mel filterbank
mel_basis = librosa.filters.mel(sr=samplerate,n_fft=n_fft,n_mels=n_mels,fmin=20,fmax=samplerate//2)
# apply filterbank on power spectrogram
mel_spectrum = np.dot(mel_basis, power_spectrogram)
logmel = librosa.power_to_db(mel_spectrum)
logmel = logmel.T # The resulted feature matrix has the shape (Time, Frequency)
save_TF_img(logmel,'logmel')

#---------
# 2. Extract the wavelet filter-bank feature
#---------
# Create wavelet filterbank
phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(np.log2(n_fft), J, Q)
wavelet_basis = [psi_f[0] for psi_f in psi1_f]
wavelet_basis = np.array(wavelet_basis); wavelet_basis = np.flip(wavelet_basis, axis=0)
wavelet_basis=np.concatenate((phi_f[0].reshape(1,-1), wavelet_basis),axis=0)
wavelet_basis = wavelet_basis[:,0:n_fft//2+1]
# apply filterbank on power spectrogram
waveletFB = np.dot(wavelet_basis, power_spectrogram)
waveletFB = librosa.power_to_db(waveletFB)
waveletFB = waveletFB.T # The resulted feature matrix has the shape (Time, Frequency)
save_TF_img(waveletFB,'waveletFB')

#---------
# 3. Decomposing wavelet filter-bank feature using temporal median miltering
#---------
median_filter_length = 101 # Suggested values are 51, 101, 201.
waveletFB_long = scipy.signal.medfilt(waveletFB, kernel_size=[median_filter_length,1])
waveletFB_short = waveletFB - waveletFB_long		
# It is suggested that we use independent embedding feature extractors for each component. (reference: https://ieeexplore.ieee.org/abstract/document/9053194)
save_TF_img(waveletFB_long,'waveletFB_long')
save_TF_img(waveletFB_short,'waveletFB_short')
