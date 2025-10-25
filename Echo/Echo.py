import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import numpy as np
import librosa
# from keras.models import load_model

SR = 22050
N_MFCC = 20


dbPath = "database2.json"


# denoiser = load_model("mfcc_denoiser.h5", compile=False)
# wave_denoiser = load_model("wave_denoiser.h5", compile=False)

# def extractFingerprintDenoiseWave(path):
#     # Load full clip at 22.05 kHz
#     y, sr = librosa.load(path, sr=SR, mono=True)
#
#     # If clip is shorter than 17 seconds, pad it
#     target_len = int(17 * SR)
#     if len(y) < target_len:
#         y = np.pad(y, (0, target_len - len(y)))
#
#     # --- Denoise in chunks (model expects fixed input length) ---
#     chunk_size = 44096
#     denoised = []
#     for i in range(0, len(y), chunk_size):
#         chunk = y[i:i + chunk_size]
#         if len(chunk) < chunk_size:
#             chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
#         chunk_in = np.expand_dims(chunk, axis=(0, -1))
#         denoised_chunk = wave_denoiser.predict(chunk_in, verbose=0).flatten()
#         denoised.append(denoised_chunk)
#
#     denoised_wave = np.concatenate(denoised)
#
#     # --- Pick random 2-second segment between 5–15 seconds ---
#     start_sec = np.random.uniform(5, 15)
#     start_sample = int(start_sec * sr)
#     end_sample = start_sample + int(2 * sr)
#
#     segment = denoised_wave[start_sample:end_sample]
#     if len(segment) < int(2 * sr):
#         segment = np.pad(segment, (0, int(2 * sr) - len(segment)))
#
#     # --- Extract MFCC fingerprint ---
#     mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
#     mfcc_vec = np.mean(mfcc, axis=1)
#     return mfcc_vec.tolist()


def loadDatabase():
    if not os.path.exists(dbPath):
        print(f"Database file not found at {dbPath}")
        database = {}
    else:
        with open(dbPath, "r") as f:
            database = json.load(f)
            print(f"Loaded {len(database)} songs from {dbPath}")
    return database


def normalizeVector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def recordAudio(filename="sample.wav", duration=3, fs=22050):
    print("Recording (stay silent for the first second)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")

    audio = np.squeeze(audio)
    audio = audio[:fs]  # First 1 second

    # print("Reducing noise...")
    # reduced = nr.reduce_noise(y=audio, y_noise=audio, sr=fs)

    wav.write(filename, fs, (audio * 32767).astype(np.int16))
    print(f"Saved to {filename}")

    time.sleep(2)
    print("Playing back the recording...")
    try:
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        print("Playback failed:", e)


def matchSong(samplePath, database):
    sampleFp = normalizeVector(np.array(extractFingerprint(samplePath)))
    bestMatch, minDistance = None, float("inf")

    for song, fps in database.items():
        for fp in fps:
            fp = np.array(fp)
            distance = np.linalg.norm(sampleFp - fp)
            # if distance < 0.2:
            #     print(f"{song}: {distance}")  # Debugging output
            if distance < minDistance:
                minDistance = distance
                bestMatch = song

    # if minDistance > 0.1:  # Threshold to reject poor matches
    #     return "No good match found"
    print(bestMatch)
    return bestMatch

def matchSongWithEmbeddings(sample_embeddings, database):
    bestMatch, minDistance = None, float("inf")
    for song, fps in database.items():  # fps = list of segment embeddings
        for fp in fps:
            fp = np.array(fp)
            for seg_emb in sample_embeddings:  # compare each segment of the sample
                distance = np.linalg.norm(seg_emb - fp)
                if distance < minDistance:
                    minDistance = distance
                    bestMatch = song
    return bestMatch

def extractFingerprint(filePath):
    off=np.random.uniform(5, 25)
    y, sr = librosa.load(filePath, sr=None, mono=True, offset=off, duration=2.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1).tolist()

if __name__ == "__main__":
    # recordAudio()
    songs = ["Compensating", "LostInJapan", "LoveAndWar", "OtherBeds", "Swervin", "WhatDoYouMean",
            "CaliforniaGurls", "ChuckTaylor", "CrazyTrain", "HowDeep", "Starlight", "TokyoDrift"]
    # songs=["WhatDoYouMean"]
    database = loadDatabase()
    if database:
        # match = matchSong("sample.wav", database)
        hitCount=[]
        repeat=100
        for song in songs:
            hitCountPer=0
            for i in range(repeat):
                match = matchSong("noise2/"+song+".wav", database)
                if str(match) == song + ".wav":
                    hitCountPer+=1
            hitCount.append(hitCountPer/repeat)
        print(hitCount)
        # print(f"Best match: {match}")
