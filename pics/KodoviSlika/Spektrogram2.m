clc, clear all, close all;
% Generate some example data (sine wave)
Fs = 1000;              % Sampling frequency
t = 0:1/Fs:1;           % Time vector
sig = sin(2*pi*50*t)+0.3*sin(2*pi*150*t)+0.1*sin(2*pi*250*t);
% [sig, Fs] = audioread('C:\Users\Mjau\Desktop\FAX\MS\ASM\Projekat\Muzika\klavir_A4.wav');
% sig=sig(:,1);

% Compute and plot the spectrogram
win = hamming(256);          % Window function
nol = 128;                 % Overlap between segments
nfft = 512;                     % Number of FFT points
[S,F,T] = spectrogram(sig, win, nol, nfft, Fs, 'yaxis');
% S: Spectrogram matrix, F: Frequency vector, T: Time vector

% Plot the spectrogram
figure;
imagesc(T, F, 10*log10(abs(S)));  % Plot the spectrogram in dB
axis xy;                         % Flip the y-axis to have zero frequency at the bottom
xlabel('Vreme (s)');
ylabel('Frekvenca (Hz)');
title('Spektrogram');
colorbar;                       % Add color bar

% Plot the 3D mesh
figure;
surf(T, F, 10*log10(abs(S)));  % Plot the 3D mesh of the spectrogram in dB
xlabel('Vreme (s)');
ylabel('Frekvenca (Hz)');
zlabel('Amplituda (dB)');
title('3D Spektrogram');