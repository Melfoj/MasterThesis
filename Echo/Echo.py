import os
import json
import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import noisereduce as nr
from scipy.signal import butter, lfilter

DB_JSON = "database.json"



def lowpassFilter(data, cutoff=8000, fs=22050, order=6):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def loadDatabase():
    if not os.path.exists(DB_JSON):
        print("Database file not found. Run build_database.py first.")
        return {}
    with open(DB_JSON, "r") as f:
        return json.load(f)

def extractFingerprint(filePath):
    y, sr = librosa.load(filePath, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).tolist()


def normalizeVector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def recordAudioOLD(duration=5, filename="sample.wav", fs=44100):
    print("Recording...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        print("Recording failed:", e)
        return

    audio = np.squeeze(audio)

    # Apply noise reduction
    print("Reducing noise...")
    reduced = nr.reduce_noise(y=audio, sr=fs)

    # Save to file
    wav.write(filename, fs, (reduced * 32767).astype(np.int16))
    print(f"Saved to {filename}")

    time.sleep(2)
    print("Playing back the recording...")
    try:
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        print("Playback failed:", e)

def recordAudio(filename="sample.wav", duration=5, fs=22050):
    print("Recording (stay silent for the first second)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")

    audio = np.squeeze(audio)
    noise_sample = audio[:fs]  # First 1 second

    print("Reducing noise...")
    reduced = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=fs)

    print("Applying low-pass filter...")
    filtered = lowpassFilter(reduced, cutoff=8000, fs=fs)

    wav.write(filename, fs, (filtered * 32767).astype(np.int16))
    print(f"Saved to {filename}")

    time.sleep(2)
    print("Playing back the recording...")
    try:
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        print("Playback failed:", e)


def matchSongWoSlide(samplePath, database):
    sampleFp = normalizeVector(np.array(extractFingerprint(samplePath)))
    bestMatch, minDistance = None, float("inf")
    for song, fp in database.items():
        fp = np.array(fp)
        distance = np.linalg.norm(sampleFp - fp)
        print(f"{song}: {distance}")  # Debugging output
        if distance < minDistance:
            minDistance, bestMatch = distance, song
    return bestMatch if bestMatch else "No match found"

def matchSong(samplePath, database):
    sampleFp = normalizeVector(np.array(extractFingerprint(samplePath)))
    bestMatch, minDistance = None, float("inf")
    for song, fps in database.items():
        for fp in fps:  # fp is a segment
            fp = np.array(fp)
            distance = np.linalg.norm(sampleFp - fp)
            if distance < 0.2:
                print(f"{song}: {distance}")  # Debugging output
            if distance < minDistance:
                minDistance = distance
                bestMatch = song
    return bestMatch


if __name__ == "__main__":
    recordAudio()
    database = loadDatabase()
    if database:
        match = matchSong("sample.wav", database)
        #match = matchSong("songs/TokyoDrift.wav", database)
        print(f"Best match: {match}")
