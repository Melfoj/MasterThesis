import os
import json
import numpy as np
import librosa

DB_PATH = "songs/"  # Folder containing reference songs
DB_JSON = "database2.json"


def extractSlidingFingerprints(filePath, segmentDuration=2.0, overlap=0.5):
    y, sr = librosa.load(filePath, sr=None)
    hop = int((1 - overlap) * segmentDuration * sr)
    win = int(segmentDuration * sr)
    fingerprints = []
    for start in range(0, len(y) - win, hop):
        segment = y[start:start+win]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        fp = np.mean(mfcc, axis=1)
        fp = normalizeVector(fp)
        fingerprints.append(fp)
    return fingerprints


def extractFingerprint(filePath):
    try:
        y, sr = librosa.load(filePath, sr=None)
        if y is None or sr is None:
            print(f"Failed to load audio: {filePath}")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        if mfcc is None or mfcc.size == 0:
            print(f"Failed to extract MFCC: {filePath}")
            return None
        return np.mean(mfcc, axis=1).tolist()
    except Exception as e:
        print(f"Error processing {filePath}: {e}")
        return None


def normalizeVector(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    return (vec if norm == 0 else vec / norm).tolist()


def buildDatabase():
    database = {}
    for file in os.listdir(DB_PATH):
        if file.endswith(".wav"):
            path = os.path.join(DB_PATH, file)
            #fingerprint = extractFingerprint(path)
            # if fingerprint is None:
            #     continue
            #normalizedFingerprint = normalizeVector(fingerprint)
            #database[file] = normalizedFingerprint
            database[file] = extractSlidingFingerprints(path)
    with open(DB_JSON, "w") as f:
        json.dump(database, f)
    print(f"Saved {len(database)} songs into {DB_JSON}.")


if __name__ == "__main__":
    buildDatabase()
