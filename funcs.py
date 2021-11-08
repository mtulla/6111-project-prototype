import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import convolve, fftconvolve, stft, resample, decimate
import wave

def from_wav(fname):
    """
    Loads the wav file passed. 
    
    Will automatically normalize the amplitude values so they are in the
    range (-1, 1). If input is stereo, will only throw out data from one side
     and return mono data.

    Parameters
        fname: path to the wav file

    Returns
        samplerate: sample rate of the wav file
        data: mono audio data as a numpy array
    """
    samplerate, sound_arr = wavfile.read(fname)
    wav_obj = wave.open(fname, mode='rb')

    if sound_arr.shape[1] > 1:
        data = sound_arr[:, 0]
    else:
        data = sound_arr
    
    return samplerate, data / 2**(wav_obj.getsampwidth() * 8 - 1)


def to_wav(fname, samplerate, data):
    """
    Writes the given audio data to a wav file.

    Will automatically detect the correct bit width.

    Parameters
        fname: file path to write wav file
        samplerate: sample rate of audio data
        data: mono audio data as a numpy array
    """
    wavfile.write(fname, samplerate, data)


def normalize(x):
    """
    Normalize the given signal so it's energy is 1

    Parameters
        x: signal as a numpy array
    
    Returns
        x_n: normalized signal that has energy of 1
    """
    normalization = (np.sum(np.square(x)))**0.5
    return x / normalization


def window_convolve(data, ir, window_size=2048):
    """
    Convolves data with ir using a windowing approach.

    Will first window data into windows of size window_size and then apply the
    convolution window by window. The results are then added together to
    produce the convolution. If the approach where being done in real-time,
    given that each window convolution has size
        nwindow_conv = window_size + len(data) - 1,
    there would need to be a buffer capable of storing
        buffer_size = nwindow_conv * ceiling(nwindow_conv / window_size)
    samples.

    Parameters:
        data: audio data as a numpy array
        ir: impulse response to convolve with as a numpy array
        window_size: window_size for convolutions
    
    Returns:
        convolution: convolved data using the procedure described as a numpy
        array
    """
    padded_data = np.pad(data, (0, -len(data) % window_size))
    nwindows = len(padded_data) // window_size
    nconvolution = len(padded_data) + len(ir) - 1
    windows = np.reshape(padded_data, (nwindows, window_size))
    convolutions = np.apply_along_axis(lambda m: fftconvolve(m, ir), 1, windows)

    convolutions_time_domain = np.empty((nwindows, nconvolution))
    for window_idx, convolution in enumerate(convolutions):
        padding = (window_size * window_idx, nconvolution - window_size * window_idx - len(convolution))
        convolutions_time_domain[window_idx] = np.pad(convolution, padding)

    return np.sum(convolutions_time_domain, axis=0)


def combine(audio1, audio2, mode='add'):
    """
    Combines the two audio samples either by adding or subtracting.

    Parameter:
        audio1: audio sample as a numpy array
        audio2: audio sample to add or subtract as a numpy array
        mode: 'add' or 'subtract'. Function returns audio1 + audio2 or
        audio1 - audio2 accordingly.

    Returns
        combined_audio: numpy array
    """
    n = max(len(audio1), len(audio2))

    if len(audio1) < n:
        if mode == 'add':
            return np.pad(audio1, (0, n - len(audio1))) + audio2 
        elif mode == 'subtract':
            return np.pad(audio1, (0, n - len(audio1))) - audio2 

    elif len(audio2) < n:
        if mode == 'add':
            return audio1 + np.pad(audio2, (0, n - len(audio2)))
        elif mode == 'subtract':
            return audio1 - np.pad(audio2, (0, n - len(audio2)))
    else:
        if mode == 'add':
            return audio1 + audio2
        elif mode == 'subtract':
            return audio1 - audio2

