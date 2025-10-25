import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

DB_PATH = "songs/"

def extractMfcc(file, duration=5.0, sr=22050):
    y, sr = librosa.load(file, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean / np.linalg.norm(mfcc_mean)

def generatePairs():
    pairs, labels = [], []
    files = [f for f in os.listdir(DB_PATH) if f.endswith(".wav")]
    for i in range(len(files)):
        for j in range(len(files)):
            mfcc1 = extractMfcc(os.path.join(DB_PATH, files[i]))
            mfcc2 = extractMfcc(os.path.join(DB_PATH, files[j]))
            pairs.append([mfcc1, mfcc2])
            labels.append(1 if i == j else 0)
    return np.array(pairs), np.array(labels)

def buildEmbedder(input_dim=20, embed_dim=32):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(embed_dim)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return models.Model(inputs, outputs)

def buildSiamese(embedder):
    inputA = layers.Input(shape=(20,))
    inputB = layers.Input(shape=(20,))
    embA = embedder(inputA)
    embB = embedder(inputB)
    distance = layers.Lambda(lambda tensors: tf.norm(tensors[0] - tensors[1], axis=1, keepdims=True))([embA, embB])
    model = models.Model([inputA, inputB], distance)
    return model

if __name__ == "__main__":
    pairs, labels = generatePairs()
    dataA, dataB = pairs[:,0,:], pairs[:,1,:]

    embedder = buildEmbedder()
    siamese = buildSiamese(embedder)

    siamese.compile(optimizer=optimizers.Adam(1e-3), loss='mse')

    siamese.fit([dataA, dataB], labels, epochs=50, batch_size=8)

    embedder.save("audio_embedder_model.h5")
    print("Saved embedding model.")
