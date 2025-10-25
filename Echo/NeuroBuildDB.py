import os
import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

DB_PATH = "songs/"
DB_JSON = "database.json"
SEGMENT_DURATION = 3.0
OVERLAP = 0.5
SR = 22050

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def extract_segments(file_path, segment_duration=SEGMENT_DURATION, overlap=OVERLAP):
    y, sr = librosa.load(file_path, sr=SR)
    hop = int((1 - overlap) * segment_duration * sr)
    win = int(segment_duration * sr)
    segments = []
    for start in range(0, len(y) - win, hop):
        seg = y[start:start + win]
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
        fp = np.mean(mfcc, axis=1)
        fp = normalize_vector(fp)
        segments.append(fp)
    return np.array(segments)

def build_database():
    base_network = load_model("base_network.keras", compile=False, custom_objects={'euclidean_distance': euclidean_distance})

    database = {}
    for file in os.listdir(DB_PATH):
        if file.endswith(".wav"):
            path = os.path.join(DB_PATH, file)
            segments = extract_segments(path)
            embeddings = base_network.predict(segments, verbose=0)
            database[file] = embeddings.tolist()

    with open(DB_JSON, "w") as f:
        json.dump(database, f)

    print(f"Saved database with {len(database)} songs.")

if __name__ == "__main__":
    import tensorflow as tf
    build_database()
