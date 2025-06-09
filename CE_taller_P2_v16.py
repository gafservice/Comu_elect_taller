# === Importación de librerías necesarias ===
import tkinter as tk                         # Para crear la interfaz gráfica (ventanas, botones, etc.)
from PIL import Image, ImageTk               # Para manejar y mostrar imágenes en la interfaz
import sounddevice as sd                     # Para grabar y reproducir audio
import numpy as np                           # Para operaciones numéricas con vectores y matrices
from scipy.io.wavfile import write, read     # Para guardar y leer archivos de audio en formato WAV
from scipy.signal import hilbert             # Para obtener la transformada de Hilbert (para modulación SSB)
import os                                    # Para operaciones con archivos
import threading                             # Para ejecutar tareas en paralelo sin bloquear la GUI
import matplotlib.pyplot as plt              # Para graficar señales en el tiempo y frecuencia
import time                                  # Para temporizar acciones como pausas cortas

# === Parámetros generales de configuración de audio ===
fs = 44100                                   # Frecuencia de muestreo estándar (44.1 kHz)
duracion = 5                                 # Duración de la grabación en segundos
archivo_baja = 'audio_baja.wav'              # Archivo para grabar el canal bajo (LSB)
archivo_alta = 'audio_alta.wav'              # Archivo para grabar el canal alto (USB)
archivo_L_ISB = 'audio_baja.wav'             # Canal izquierdo en modulación ISB
archivo_R_ISB = 'audio_alta.wav'             # Canal derecho en modulación ISB

# === Parámetros específicos para modulación ===
FS = 44100                                   # Frecuencia de muestreo utilizada internamente
FC = 10000                                   # Frecuencia de la portadora (10 kHz)
DUR_TONO = 0.4                               # Duración de los tonos de inicio y fin (segundos)
TONO_INICIO = 4000                           # Frecuencia del tono de inicio (Hz)
TONO_FIN = 5000                              # Frecuencia del tono de fin (Hz)

# === Función para suavizar una señal de audio con media móvil ===
def suavizar(audio, N=5):
    return np.convolve(audio, np.ones(N)/N, mode='same').astype(np.int16)

# === Función para generar un tono sinusoidal de frecuencia y duración específicas ===
def generar_tono(freq, duracion, fs):
    t = np.arange(int(fs * duracion)) / fs
    A = 0.7
    return A * np.sin(2 * np.pi * freq * t)

# === Función para reproducir una señal de audio ===
def reproducir_senal(senal, fs):
    sd.play(senal, fs)
    sd.wait()

# === Cargar archivo de audio, convertir a mono si es estéreo y normalizar ===
def cargar_audio(nombre_archivo):
    fs, audio = read(nombre_archivo)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # Promediar canales si es estéreo
    audio /= np.max(np.abs(audio))  # Normalización
    return fs, audio

# === Función para grabar audio y guardar en archivo WAV ===
def grabar_audio(nombre_archivo):
    def grabar():
        estado_var.set(f"🎙️ Grabando en {nombre_archivo}...")
        root.update()
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = audio[:, 0]  # Convertir a un solo canal
        audio = audio - np.mean(audio)  # Quitar componente DC
        audio = audio / np.max(np.abs(audio))  # Normalizar a [-1, 1]
        audio = audio * 0.95  # Dejar un pequeño margen para evitar recortes
        audio_int16 = (audio * 32767).astype(np.int16)  # Convertir a formato 16-bit
        write(nombre_archivo, fs, suavizar(audio_int16))  # Guardar suavizado
        estado_var.set(f"✅ Guardado: {nombre_archivo}")
        root.update()
    threading.Thread(target=grabar).start()

# === Reproducir un archivo WAV si existe ===
def reproducir_audio(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        estado_var.set(f"❌ Archivo no encontrado: {nombre_archivo}")
        return
    fs_leido, datos = read(nombre_archivo)
    reproducir_senal(datos, fs_leido)
    estado_var.set(f"✅ Reproducción: {nombre_archivo}")

# === Función para graficar señal en tiempo y frecuencia ===
def graficar_senal_tiempo_frecuencia(senal, fs, titulo, usar_analitica=False, max_magnitud=100):
    t = np.arange(len(senal)) / fs
    plt.figure(figsize=(12, 4))

    # Gráfica en el tiempo
    plt.subplot(1, 2, 1)
    if np.iscomplexobj(senal):
        plt.plot(t, np.real(senal), label='Real')
        plt.plot(t, np.imag(senal), label='Imag', alpha=0.5)
        plt.legend()
    else:
        plt.plot(t, senal)
    plt.title(f'{titulo} - Tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Gráfica en frecuencia
    plt.subplot(1, 2, 2)
    if usar_analitica and not np.iscomplexobj(senal):
        senal = hilbert(senal)
    N = len(senal)
    espectro = np.abs(np.fft.fft(senal))
    freqs = np.fft.fftfreq(N, 1/fs)
    indices = np.where((freqs >= 0) & (freqs <= 20000))
    espectro = np.clip(espectro[indices], 0, max_magnitud)
    f

