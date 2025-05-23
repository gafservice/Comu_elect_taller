import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy.signal import hilbert
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, messagebox

fs = 44100  # Sample rate
duracion = 5  # Recording duration in seconds
archivo_mono = 'audio_mono.wav'

# Global variable to control timer
recording_time_left = duracion

def actualizar_timer():
    global recording_time_left
    if recording_time_left > 0:
        timer_label.config(text=f"Grabando... {recording_time_left} s")
        recording_time_left -= 1
        root.after(1000, actualizar_timer)  # Call again after 1 second
    else:
        timer_label.config(text="Grabación finalizada.")

def grabar_audio():
    global recording_time_left
    try:
        dispositivo = None
        canales = 1
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] >= 1:
                dispositivo = i
                canales = d['max_input_channels']
                break

        if dispositivo is None:
            messagebox.showerror("Error", "No se encontró un dispositivo de entrada válido.")
            return
        
        recording_time_left = duracion
        # Start timer updates in main thread using after()
        root.after(0, actualizar_timer)

        # Record audio continuously for duracion seconds
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=canales, dtype='int16', device=dispositivo)
        sd.wait()

        timer_label.config(text="")  # Clear timer

        if canales > 1:
            audio = np.mean(audio, axis=1).astype(np.int16)
        write(archivo_mono, fs, audio)
        messagebox.showinfo("Grabación", f"Grabación guardada como {archivo_mono}")
    except Exception as e:
        messagebox.showerror("Error de Grabación", str(e))
        timer_label.config(text="")  # Clear timer in case of error

def reproducir_audio(filename):
    try:
        if not os.path.exists(filename):
            messagebox.showwarning("Archivo no encontrado", f"No se encontró el archivo {filename}")
            return
        fs, audio = read(filename)
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        messagebox.showerror("Error de Reproducción", str(e))

def tono(frecuencia, duracion):
    t = np.arange(int(fs * duracion)) / fs
    return 0.8 * np.sin(2 * np.pi * frecuencia * t)

def transmitir_audio():
    try:
        if not os.path.exists(archivo_mono):
            messagebox.showwarning("Archivo no encontrado", f"No se encontró el archivo {archivo_mono}. Grabe audio primero.")
            return

        fs, audio = read(archivo_mono)
        audio = audio.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if np.max(np.abs(audio)) != 0:
            audio /= np.max(np.abs(audio))
        else:
            messagebox.showwarning("Audio Silencioso", "El archivo de audio está en silencio.")
            return

        N = len(audio)
        t = np.arange(N) / fs
        fc = 10000
        analytic = hilbert(audio)
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))

        tono_inicio = tono(13000, 0.7)
        tono_final = tono(3000, 0.5)
        retardo = np.zeros(int(0.2 * fs))
        senal_tx = np.concatenate((retardo, tono_inicio, ssb, tono_final))

        sd.play(senal_tx, fs)
        sd.wait()
    except Exception as e:
        messagebox.showerror("Error de Transmisión", str(e))

def grabar_thread():
    threading.Thread(target=grabar_audio, daemon=True).start()

def reproducir_thread():
    threading.Thread(target=reproducir_audio, args=(archivo_mono,), daemon=True).start()

def transmitir_thread():
    threading.Thread(target=transmitir_audio, daemon=True).start()

# GUI setup
root = Tk()
root.title("Transmisión de Audio AM-SSB")

Label(root, text="Grabación y Transmisión de Audio").pack(pady=10)
Button(root, text="🎙️ Grabar 5 segundos", command=grabar_thread, width=25).pack(pady=5)
Button(root, text="▶️ Reproducir Audio Original", command=reproducir_thread, width=25).pack(pady=5)
Button(root, text="📡 Transmitir Audio Modulado", command=transmitir_thread, width=25).pack(pady=5)

timer_label = Label(root, text="", fg="red", font=("Helvetica", 12))
timer_label.pack(pady=10)

root.mainloop()
