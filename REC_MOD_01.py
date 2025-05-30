# Modulador ISB con Interfaz Gráfica

Este código crea una interfaz gráfica en Python utilizando `tkinter` que permite seleccionar diferentes tipos de modulación (SSB-SC USB/LSB, SSB-FC USB/LSB, ISB), grabar audio, convertirlo a mono, generar señales ISB y visualizar en tiempo y frecuencia.

```python
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

fc = 10000

# Funciones auxiliares
def tono(frec, dur, fs):
    t = np.arange(0, int(fs * dur)) / fs
    return 0.7 * np.sin(2 * np.pi * frec * t)

def graficar_espectro(x, fs, titulo):
    N = len(x)
    f = np.fft.fftfreq(N, d=1/fs)
    X = np.abs(np.fft.fft(x))
    X = np.clip(X, 0, 50)
    plt.figure()
    plt.plot(f[:N//2], X[:N//2])
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.show()

def graficar_tiempo(x, fs, titulo):
    t = np.arange(len(x)) / fs
    plt.figure()
    plt.plot(t[:int(fs*0.005)], x[:int(fs*0.005)])
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

def graficar_espectro_isb(lsb, usb, fs):
    N = len(lsb)
    f = np.fft.fftfreq(N, d=1/fs)
    S_lsb = np.abs(np.fft.fft(lsb))
    S_usb = np.abs(np.fft.fft(usb))
    S_lsb = np.clip(S_lsb, 0, 50)
    S_usb = np.clip(S_usb, 0, 50)
    plt.figure()
    plt.plot(f[:N//2], S_lsb[:N//2], label="LSB (audio_L_ISB)", linestyle='--')
    plt.plot(f[:N//2], S_usb[:N//2], label="USB (audio_R_ISB)", linestyle='-')
    plt.title("Espectro combinado de ISB")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.legend()
    plt.grid(True)
    plt.show()

def cargar_audio_mono():
    fs, audio = wavfile.read("audio_mono.wav")
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio -= np.mean(audio)
    audio /= np.max(np.abs(audio))
    return fs, audio

def modulacion_ssb(audio, fs, banda='USB'):
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    return ssb

def modulacion_ssb_fc(audio, fs, banda='USB'):
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    carrier = np.cos(2 * np.pi * fc * t)
    return carrier + ssb

def modulacion_isb():
    fs_L, audio_L = wavfile.read("audio_L_ISB.wav")
    fs_R, audio_R = wavfile.read("audio_R_ISB.wav")
    assert fs_L == fs_R
    fs = fs_L
    audio_L = audio_L.astype(np.float32)
    audio_R = audio_R.astype(np.float32)
    if audio_L.ndim > 1:
        audio_L = audio_L.mean(axis=1)
    if audio_R.ndim > 1:
        audio_R = audio_R.mean(axis=1)
    audio_L -= np.mean(audio_L)
    audio_R -= np.mean(audio_R)
    audio_L /= np.max(np.abs(audio_L))
    audio_R /= np.max(np.abs(audio_R))
    N = min(len(audio_L), len(audio_R))
    t = np.arange(N) / fs
    analytic_L = hilbert(audio_L[:N])
    analytic_R = hilbert(audio_R[:N])
    lsb = np.real(np.conj(analytic_L) * np.exp(1j * 2 * np.pi * fc * t))
    usb = np.real(analytic_R * np.exp(1j * 2 * np.pi * fc * t))
    isb = lsb + usb
    return isb, fs, lsb, usb

# Interfaz gráfica
root = tk.Tk()
root.title("Modulador AM Avanzado")
root.geometry("400x300")

frame = ttk.Frame(root, padding=20)
frame.pack(expand=True)

def ejecutar_opcion(opcion):
    if opcion in ['SSB-SC USB', 'SSB-SC LSB', 'SSB-FC USB', 'SSB-FC LSB']:
        fs, audio = cargar_audio_mono()
        banda = 'USB' if 'USB' in opcion else 'LSB'
        if 'FC' in opcion:
            resultado = modulacion_ssb_fc(audio, fs, banda)
        else:
            resultado = modulacion_ssb(audio, fs, banda)
        sd.play(np.concatenate((tono(7000, 0.2, fs), resultado, tono(5000, 0.2, fs))), fs, blocking=True)
        graficar_tiempo(resultado, fs, f"{opcion} (tiempo)")
        graficar_espectro(resultado, fs, f"{opcion} (frecuencia)")

    elif opcion == 'ISB':
        isb, fs, lsb, usb = modulacion_isb()
        sd.play(np.concatenate((tono(7000, 0.2, fs), isb, tono(5000, 0.2, fs))), fs, blocking=True)
        graficar_tiempo(isb, fs, "ISB (tiempo)")
        graficar_espectro(isb, fs, "ISB (frecuencia total)")
        graficar_espectro_isb(lsb, usb, fs)

for text in ['SSB-SC USB', 'SSB-SC LSB', 'SSB-FC USB', 'SSB-FC LSB', 'ISB']:
    b = ttk.Button(frame, text=text, command=lambda t=text: ejecutar_opcion(t))
    b.pack(pady=5, fill='x')

root.mainloop()
```

Este código asume que los archivos `audio_mono.wav`, `audio_L_ISB.wav` y `audio_R_ISB.wav` están disponibles en el mismo directorio. Usa `matplotlib` para mostrar los gráficos, `scipy.signal.hilbert` para obtener la señal analítica, y `sounddevice` para reproducción de audio. Se puede ampliar fácilmente con opciones para grabar, guardar y aplicar filtros si se desea.

