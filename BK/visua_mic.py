
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import get_window

fs = 44100
duration = 0.1  # ventana de 100 ms
blocksize = int(fs * duration)
window = get_window("hann", blocksize)

fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6))

# Gráfica de tiempo
line_time, = ax_time.plot(np.zeros(blocksize))
ax_time.set_title("Señal de micrófono (tiempo)")
ax_time.set_xlabel("Muestras")
ax_time.set_ylabel("Amplitud")
ax_time.set_ylim(-1, 1)
ax_time.grid()

# Gráfica de frecuencia (en dB)
f = np.fft.rfftfreq(blocksize, d=1/fs)
line_freq, = ax_freq.plot(f, np.zeros(len(f)))
ax_freq.set_title("Espectro (frecuencia en dB)")
ax_freq.set_xlabel("Frecuencia [Hz]")
ax_freq.set_ylabel("Magnitud [dB]")
ax_freq.set_ylim(-100, 0)
ax_freq.grid()

plt.tight_layout()
plt.ion()
plt.show()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # Señal en el tiempo
    muestra = indata[:, 0]
    line_time.set_ydata(muestra)

    # Magnitud espectral en dB
    spectrum = np.fft.rfft(muestra * window)
    magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)
    line_freq.set_ydata(magnitude)

    fig.canvas.draw()
    fig.canvas.flush_events()

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=blocksize):
        print("🎙 Visualizando en tiempo real... Presioná Ctrl+C para salir.")
        while True:
            plt.pause(0.01)
except KeyboardInterrupt:
    print("🛑 Visualización detenida.")
