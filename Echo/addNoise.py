import numpy as np
import librosa
import soundfile as sf

songs = ["Compensating", "LostInJapan", "LoveAndWar", "OtherBeds", "Swervin", "WhatDoYouMean", "CaliforniaGurls", "ChuckTaylor", "CrazyTrain", "HowDeep", "Starlight", "TokyoDrift"]

for song in songs:
    # Load wav file
    y, sr = librosa.load("songs/"+song+".wav", sr=None)  # y = audio time series, sr = sample rate

    # Function to add Gaussian noise
    def add_noise(y, noise_factor=0.005):
        noise = np.random.randn(len(y))   # generate random noise
        y_noisy = y + noise_factor * noise
        return np.clip(y_noisy, -1.0, 1.0)  # keep values in [-1, 1]

    # Apply noise
    y_noisy = add_noise(y, noise_factor=0.1)

    # Save to new file
    sf.write("noise3/"+song+".wav", y_noisy, sr)
