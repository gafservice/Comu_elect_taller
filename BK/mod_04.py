import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# === Lectura del audio ===
fs, audio = wavfile.read("audio_mono.wav")
audio = audio.astype(np.float32)

# Convertir a mono si es estéreo
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Normalizar
audio /= np.max(np.abs(audio))
N = len(audio)
t = np.arange(N) / fs

# === Modulación AM-SSB ===
fc = 10000  # Hz
analytic = hilbert(audio)
ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))

# === Tono de inicio (2000 Hz) y fin (3000 Hz) ===
dur_tono = 0.2  # segundos
t_tono = np.arange(0, int(fs * dur_tono)) / fs
tono_inicio = 0.7 * np.sin(2 * np.pi * 2000 * t_tono)
tono_fin = 0.7 * np.sin(2 * np.pi * 3000 * t_tono)

# === Concatenar señal total a reproducir ===
señal_total = np.concatenate((tono_inicio, ssb, tono_fin))

# === Transmisión ===
print("Reproduciendo señal AM-SSB (con tono de inicio y fin)...")
sd.play(señal_total, fs, blocking=True, blocksize=1024)
sd.wait()
print("Transmisión finalizada.")

# === FFTs para análisis ===
f_audio = np.fft.fftfreq(N, d=1/fs)
S_audio_mag = np.abs(np.fft.fft(audio))

f_ssb = np.fft.fftfreq(N, d=1/fs)
S_ssb_mag = np.abs(np.fft.fft(ssb))

# === Gráfica 1: Audio original en el tiempo ===
plt.figure()
plt.plot(t[:int(fs*0.005)], audio[:int(fs*0.005)])
plt.title("Audio original (tiempo, 5 ms)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.show(block=False)

# === Gráfica 2: Espectro del audio original ===
plt.figure()
plt.plot(f_audio[:N//2], S_audio_mag[:N//2])
plt.title("Espectro del audio original (banda base)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.show(block=False)

# === Gráfica 3: Señal AM-SSB en el tiempo ===
plt.figure()
plt.plot(t[:int(fs*0.005)], ssb[:int(fs*0.005)])
plt.title("Señal AM-SSB (tiempo, 5 ms)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.show(block=False)

# === Gráfica 4: Espectro de la señal AM-SSB ===
plt.figure()
plt.plot(f_ssb[:N//2], S_ssb_mag[:N//2])
plt.title("Espectro de la señal AM-SSB (centrado en 10 kHz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.show(block=False)

input("✅ Presione Enter para cerrar todas las gráficas...")

