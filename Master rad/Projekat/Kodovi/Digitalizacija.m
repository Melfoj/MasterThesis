clc;clear all;close all;

Fs = 100; % Sampling frequency (Hz)
t = -5:1/Fs:5; % Time vector from -5 to 5 seconds
f = 0.24; % Frequency of the sine wave (Hz)

% Signal
sig = sin(2*pi*f*t);

% Digitalization
bit = 4; % Number of bits for quantization
quantl = 2^bit; % Number of quantization levels
quants = 2/(quantl-1); % Quantization step size
digsig = round(sig / quants) * quants;
for i=2:2:length(digsig)
    digsig(i)=digsig(i-1);
end

% Plot
figure;
plot(t, sig, 'b', 'LineWidth', 2);
hold on;
plot(t, digsig, 'r', 'LineWidth', 2);
title('Digitalizacija signala');
xlabel('Vreme (s)');
ylabel('Amplituda');
legend('Analogni', 'Digitalni');
grid on;
