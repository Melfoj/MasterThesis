import os
import librosa
import numpy as np
import soundfile as sf
from librosa.effects import pitch_shift


def add_noise(y, noise_factor=0.005):
    y = np.asarray(y)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # convert stereo to mono if needed
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(y.dtype)

def process_and_save(file_path, output_dir="augmented_songs", num_augments=5):
    y, sr = librosa.load(file_path, sr=None)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save original file with a consistent naming convention
    original_path = os.path.join(output_dir, f"{base_name}_orig.wav")
    sf.write(original_path, y, sr)

    # Generate augmented versions
    for i in range(num_augments):
        # Random pitch shift between -5 and 5 semitones
        pitch_amt = np.random.uniform(-5, 5)
        y_shifted = pitch_shift(y,sr= sr, n_steps=pitch_amt)

        # Add random noise with a random factor between 0.001 and 0.01
        noise_factor = np.random.uniform(0.001, 0.01)
        y_augmented = add_noise(y_shifted, noise_factor)

        # Save augmented file
        augmented_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.wav")
        sf.write(augmented_path, y_augmented, sr)
        print(f"Saved augmented file: {augmented_path}")

def main():
    input_folder = "songs"  # Your original songs folder
    output_folder = "augmented_songs"  # Where to save augmented files

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(input_folder, file)
            process_and_save(file_path, output_dir=output_folder)

if __name__ == "__main__":
    main()
