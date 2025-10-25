import numpy as np
import librosa
import json
from tensorflow.keras.models import load_model

SR = 22050
N_MFCC = 20
SEGMENT_DURATION = 3.0
OVERLAP = 0.5


denoiser = load_model("mfcc_denoiser.h5")
base_network = load_model("base_network.keras")  # your embedding model


db_filename = r"/Echo/noise2/my_database.db"
with open(db_filename, "r") as f:
    database = json.load(f)

print(f"Loaded {len(database)} songs from {db_filename}")


def extract_segments(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    hop = int((1 - OVERLAP) * SEGMENT_DURATION * sr)
    win = int(SEGMENT_DURATION * sr)
    segments = []
    for start in range(0, len(y) - win, hop):
        seg = y[start:start + win]
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC)
        segments.append(np.mean(mfcc, axis=1))
    return np.array(segments)

def get_embeddings(file_path):
    segments = extract_segments(file_path)
    segments = denoiser.predict(segments)       # denoise
    embeddings = base_network.predict(segments) # get embeddings
    return embeddings

