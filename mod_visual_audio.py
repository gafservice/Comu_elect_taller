
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# === Lectura del archivo WAV ===
fs, audio = wavfile.read("audio_mono.wav")
audio = audio.astype(np.float32)

# Convertir a mono si es estéreo
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Normalizar
if np.max(np.abs(audio)) != 0:
    audio /= np.max(np.abs(audio))
else:
    print("⚠️ El archivo de audio está en silencio.")

N = len(audio)
t = np.arange(N) / fs

# === Reproducir señal original ===
print("🎧 Reproduciendo señal original...")
sd.play(audio, fs)
sd.wait()
print("✅ Reproducción de entrada completada.")

# === Modulación AM-SSB ===
fc = 10000  # Hz
analytic = hilbert(audio)
ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))

# === Tonos de inicio y fin ===
def tono(frecuencia, duracion):
    t = np.arange(int(fs * duracion)) / fs
    return 0.8 * np.sin(2 * np.pi * frecuencia * t)

tono_inicio = tono(5000, 0.7)
tono_final = tono(3000, 0.5)
retardo = np.zeros(int(0.2 * fs))  # 0.2 s de silencio

# === Señal final: retardo + tono de inicio + ssb + tono de fin ===
senal_tx = np.concatenate((retardo, tono_inicio, ssb, tono_final))

# === Reproducción de la señal modulada ===
print("📡 Reproduciendo señal AM-SSB con tonos de inicio y fin...")
sd.play(senal_tx, fs, blocking=True, blocksize=1024)
sd.wait()
print("✅ Transmisión finalizada.")

# === FFT ===
def compute_fft(signal, fs):
    N = len(signal)
    f = np.fft.fftfreq(N, d=1/fs)
    S = np.abs(np.fft.fft(signal))
    return f[:N//2], S[:N//2]

f_audio, S_audio = compute_fft(audio, fs)
f_ssb, S_ssb = compute_fft(ssb, fs)

# === Visualización ===
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Seleccionar segmento con energía (0.2 s en adelante)
inicio = int(fs * 0.2)
fin = inicio + int(fs * 0.01)
t_seg = t[inicio:fin]
audio_seg = audio[inicio:fin]
ssb_seg = ssb[inicio:fin]

axs[0, 0].plot(t_seg, audio_seg)
axs[0, 0].set_title("Audio reproducido (tiempo)")
axs[0, 0].set_xlabel("Tiempo [s]")
axs[0, 0].set_ylabel("Amplitud")
axs[0, 0].grid(True)

axs[0, 1].plot(t_seg, ssb_seg)
axs[0, 1].set_title("Señal modulada AM-SSB (tiempo)")
axs[0, 1].set_xlabel("Tiempo [s]")
axs[0, 1].set_ylabel("Amplitud")
axs[0, 1].grid(True)

axs[1, 0].plot(f_audio, S_audio)
axs[1, 0].set_title("Señal de entrada (frecuencia)")
axs[1, 0].set_xlabel("Frecuencia [Hz]")
axs[1, 0].set_ylabel("Magnitud")
axs[1, 0].grid(True)

axs[1, 1].plot(f_ssb, S_ssb)
axs[1, 1].set_title("Señal modulada AM-SSB (frecuencia)")
axs[1, 1].set_xlabel("Frecuencia [Hz]")
axs[1, 1].set_ylabel("Magnitud")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
