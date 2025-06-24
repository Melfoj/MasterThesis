import pyaudio
import keyboard
import wave
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from scipy.signal import butter, lfilter

def lowpassFilter(data, cutoff=8000, fs=22050, order=6):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

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



def recordSong():
    chunk = 1024
    form = pyaudio.paInt16
    channels = 2
    rate = 44100
    Output_Filename = "sample.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=form,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    print("Press SPACE to start recording")
    keyboard.wait('space')
    print("Recording... Press SPACE to stop.")
    time.sleep(0.2)

    while True:
        try:
            data = stream.read(chunk)
            frames.append(data)
        except KeyboardInterrupt:
            break
        if keyboard.is_pressed('space'):
            print("stopping recording")
            time.sleep(0.2)
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(Output_Filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(form))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


recordAudio()

