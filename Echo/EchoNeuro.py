import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

DB_JSON = "database.json"
MODEL_PATH = "base_network.keras"
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

def load_database():
    with open(DB_JSON, "r") as f:
        return json.load(f)

def find_best_match(file_path):
    base_network = load_model(MODEL_PATH, compile=False, custom_objects={'euclidean_distance': euclidean_distance})
    database = load_database()

    segments = extract_segments(file_path)
    query_embeddings = base_network.predict(segments, verbose=0)

    best_match = None
    best_distance = float('inf')

    for song, embeddings in database.items():
        db_embeddings = np.array(embeddings)
        for q_emb in query_embeddings:
            distances = np.linalg.norm(db_embeddings - q_emb, axis=1)
            min_dist = np.min(distances)
            if min_dist < best_distance:
                best_distance = min_dist
                best_match = song

    print(f"Best match: {best_match} with distance {best_distance:.4f}")
    return best_match

if __name__ == "__main__":
    import tensorflow as tf
    # Example usage:
    find_best_match("sample.wav")
