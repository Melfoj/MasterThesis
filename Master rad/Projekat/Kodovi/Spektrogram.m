clc, clear all, close all;
% Load the audio file
[sig, Fs] = audioread('C:\Users\Mjau\Desktop\FAX\MS\ASM\Projekat\Muzika\klavir_A4.wav');
sig=sig(:,1);

% cutoff_frequency = 4000; % Hz
% [b, a] = butter(6, cutoff_frequency/(Fs/2), 'low'); % 6th order Butterworth filter
% y = filtfilt(b, a, y);

% Compute the spectrogram
windowSize = 1024;
overlap = 512;
nfft = 1024;
[s, f, t] = spectrogram(sig, hamming(windowSize), overlap, nfft, Fs);
figure, spectrogram(sig,'yaxis');

% Plot the spectrogram
figure;
imagesc(t, f, 10*log10(abs(s)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');

% Find peaks in the spectrogram for each time frame
threshold = 15; % Adjust this threshold as needed
peaks = [];
peak_frequencies = [];
peak_times = [];
for i = 1:size(s,2)
    [pks, locs] = findpeaks(10*log10(abs(s(:,i))), 'MinPeakHeight', threshold, 'MinPeakDistance', 10);
    peaks = [peaks; pks];
    peak_frequencies = [peak_frequencies; f(locs)];
    peak_times = [peak_times; t(i)*ones(size(pks))];
end

% Plot peaks on the spectrogram
hold on;
plot(peak_times, peak_frequencies, 'r.', 'MarkerSize', 10);
hold off;