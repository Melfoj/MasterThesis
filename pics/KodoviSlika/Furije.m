clc;clear all;close all;

Fs = 1000;          % Sampling frequency (Hz)
T = 1;              % Duration of signal (s)
t = 0:1/Fs:T-1/Fs;  % Time vector

% Define frequencies and amplitudes of sinusoids
freq =10*(1:2:13);    % Frequencies of sinusoids (Hz)
amp = 4./(1:2:13);      % Amplitudes of sinusoids

% Generate signal as sum of sinusoids
sig = zeros(size(t));
for i = 1:length(freq)
    sig = sig + amp(i) * sin(2 * pi * freq(i) * t);
end

% Perform FFT
N = length(sig);             % Length of signal
f = Fs*(0:(N/2))/N;             % Frequency vector for plotting
fft_signal = fft(sig)/N;     % Compute FFT and normalize

% Plot signal
subplot(2, 1, 1);
plot(t, sig);
xlim([0 0.2]);
title('Originalni signal');
xlabel('Vreme (s)');
ylabel('Amplituda');

% Plot FFT
subplot(2, 1, 2);
plot(f, 2*abs(fft_signal(1:N/2+1)));
title('FFT signala');
xlabel('Frekvenca (Hz)');
ylabel('Amplituda');

% Adjust plot
xlim([0, max(freq)*3]);  % Limit x-axis to maximum frequency of sinusoids
grid on;
