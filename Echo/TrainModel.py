import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

PAIR_SAVE_PATH = "training_pairs.npz"
EMBED_DIM = 16

@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, keras.backend.epsilon()))

def create_base_network(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(EMBED_DIM)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)  # normalize embeddings
    return Model(inputs, x)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def train():
    data = np.load(PAIR_SAVE_PATH, allow_pickle=True)
    pairs = data['pairs']
    labels = data['labels']

    x1 = np.stack(pairs[:, 0])
    x2 = np.stack(pairs[:, 1])
    y = labels.astype(np.float32)

    input_shape = x1.shape[1:]  # e.g. (20,)

    base_network = create_base_network(input_shape)

    input_a = keras.Input(shape=input_shape)
    input_b = keras.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])

    model.fit([x1, x2], y, batch_size=32, epochs=50, validation_split=0.1)

    model.save("siamese_model.keras")
    base_network.save("base_network.keras")
    print("Training complete and models saved.")

if __name__ == "__main__":
    train()
