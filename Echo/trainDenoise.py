import os
import numpy as np
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D
from tensorflow.keras.optimizers import Adam

# === Parameters ===
SR = 22050
DURATION = 2.0  # seconds
SAMPLES = int(SR * DURATION)
# Make SAMPLES divisible by 8 to avoid rounding mismatch
SAMPLES = (SAMPLES // 8) * 8

EPOCHS = 500
BATCH_SIZE = 16

NOISE_DIRS = ["noise2", "noise3"]
CLEAN_DIR = "songs"

def load_audio_fixed(path, sr=SR, length=SAMPLES):
    """Load and normalize audio to a fixed length."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]
        y = y / np.max(np.abs(y) + 1e-8)
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# === Build dataset ===
X_noisy, X_clean = [], []

print("\nPairing noisy and clean audio files...\n")

for noise_dir in NOISE_DIRS:
    for fname in os.listdir(noise_dir):
        if not fname.lower().endswith(".wav"):
            continue

        noisy_path = os.path.join(noise_dir, fname)
        clean_path = os.path.join(CLEAN_DIR, fname)

        if not os.path.exists(clean_path):
            print(f"No matching clean file for {noisy_path}")
            continue

        noisy_y = load_audio_fixed(noisy_path)
        clean_y = load_audio_fixed(clean_path)

        if noisy_y is not None and clean_y is not None:
            X_noisy.append(noisy_y)
            X_clean.append(clean_y)
            print(f"{noisy_path}  →  {clean_path}")
        else:
            print(f"Skipped {fname} due to loading error")

if not X_noisy:
    raise ValueError("No valid training data found.")

X_noisy = np.expand_dims(np.array(X_noisy), axis=-1)
X_clean = np.expand_dims(np.array(X_clean), axis=-1)

print(f"\nLoaded {len(X_noisy)} pairs, shape: {X_noisy.shape}")

# === Build Conv1D Autoencoder ===
inp = Input(shape=(SAMPLES, 1))

# Encoder
x = Conv1D(32, 15, activation="relu", padding="same")(inp)
x = MaxPooling1D(2, padding="same")(x)
x = Conv1D(64, 15, activation="relu", padding="same")(x)
x = MaxPooling1D(2, padding="same")(x)
x = Conv1D(128, 15, activation="relu", padding="same")(x)
encoded = MaxPooling1D(2, padding="same")(x)

# Decoder
x = Conv1D(128, 15, activation="relu", padding="same")(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(64, 15, activation="relu", padding="same")(x)
x = UpSampling1D(2)(x)
x = Conv1D(32, 15, activation="relu", padding="same")(x)
x = UpSampling1D(2)(x)

# Crop extra samples to ensure exact match
x = Cropping1D(((x.shape[1] - SAMPLES) // 2, (x.shape[1] - SAMPLES + 1) // 2))(x)

out = Conv1D(1, 15, activation="tanh", padding="same")(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=Adam(1e-4), loss="mse")

autoencoder.summary()

autoencoder.fit(
    X_noisy, X_clean,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

autoencoder.save("wave_denoiser.h5")
print("\nModel saved as wave_denoiser.h5")
