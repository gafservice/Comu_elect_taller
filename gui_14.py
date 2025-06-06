import tkinter as tk
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import hilbert
import os
import threading
import matplotlib.pyplot as plt

# === Parámetros ===
fs = 44100
duracion = 5
archivo_baja = 'audio_baja.wav'
archivo_alta = 'audio_alta.wav'
archivo_L_ISB = 'audio_baja.wav'
archivo_R_ISB = 'audio_alta.wav'

# === Parámetros de modulación ===
FS = 44100
FC = 10000
DUR_TONO = 0.2
TONO_INICIO = 7000
TONO_FIN = 5000

# === Funciones auxiliares ===
def suavizar(audio, N=5):
    return np.convolve(audio, np.ones(N)/N, mode='same').astype(np.int16)

def generar_tono(freq, duracion, fs):
    t = np.arange(int(fs * duracion)) / fs
    A = 0.7
    return A * np.sin(2 * np.pi * freq * t)

def reproducir_senal(senal, fs):
    sd.play(senal, fs)
    sd.wait()

def cargar_audio(nombre_archivo):
    fs, audio = read(nombre_archivo)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio /= np.max(np.abs(audio))
    return fs, audio

def grabar_audio(nombre_archivo):
    def grabar():
        estado_var.set(f"🎙️ Grabando en {nombre_archivo}...")
        root.update()
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = audio[:, 0]  # Convertir a 1D
        audio = audio - np.mean(audio)  # Eliminar DC
        audio = audio / np.max(np.abs(audio))  # Normalización
        audio = audio * 0.95  # Ganancia segura sin recorte
        audio_int16 = (audio * 32767).astype(np.int16)  # Convertir a int16 para guardar
        write(nombre_archivo, fs, suavizar(audio_int16))  # Puedes comentar suavizar si no se desea
        estado_var.set(f"✅ Guardado: {nombre_archivo}")
        root.update()
    threading.Thread(target=grabar).start()


def reproducir_audio(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        estado_var.set(f"❌ Archivo no encontrado: {nombre_archivo}")
        return
    fs_leido, datos = read(nombre_archivo)
    reproducir_senal(datos, fs_leido)
    estado_var.set(f"✅ Reproducción: {nombre_archivo}")

def graficar_senal_tiempo_frecuencia(senal, fs, titulo, usar_analitica=False, max_magnitud=100):
    t = np.arange(len(senal)) / fs
    plt.figure(figsize=(12, 4))
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

    plt.subplot(1, 2, 2)
    if usar_analitica and not np.iscomplexobj(senal):
        senal = hilbert(senal)
    N = len(senal)
    espectro = np.abs(np.fft.fft(senal))
    freqs = np.fft.fftfreq(N, 1/fs)
    indices = np.where((freqs >= 0) & (freqs <= 20000))
    espectro = np.clip(espectro[indices], 0, max_magnitud)
    freqs = freqs[indices]
    plt.plot(freqs, espectro)
    plt.title(f'{titulo} - Frecuencia (0–20 kHz)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.xlim([0, 20000])
    plt.ylim([0, max_magnitud + 10])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def modulacion_ssb(audio, tipo):
    t = np.arange(len(audio)) / FS
    carrier_cos = np.cos(2*np.pi*FC*t)
    carrier_sin = np.sin(2*np.pi*FC*t)
    analytic = np.imag(hilbert(audio))
    if tipo == "USB":
        return np.real(audio * carrier_cos - analytic * carrier_sin)
    else:
        return np.real(audio * carrier_cos + analytic * carrier_sin)

def modulacion_ssb_fc(audio, tipo):
    t = np.arange(len(audio)) / FS
    carrier_cos = np.cos(2*np.pi*FC*t)
    carrier_sin = np.sin(2*np.pi*FC*t)
    analytic = np.imag(hilbert(audio))
    if tipo == "USB":
        return np.real(2 * carrier_cos + (audio * carrier_cos - analytic * carrier_sin))
    else:
        return np.real(2 * carrier_cos + (audio * carrier_cos + analytic * carrier_sin))

def modulacion_isb(audio_L, audio_R):
    t = np.arange(len(audio_L)) / FS
    carrier_cos = np.cos(2*np.pi*FC*t)
    carrier_sin = np.sin(2*np.pi*FC*t)
    analyticL = np.imag(hilbert(audio_L))
    analyticR = np.imag(hilbert(audio_R))
    isb_usb = np.real(audio_L * carrier_cos - analyticL * carrier_sin)
    isb_lsb = np.real(audio_R * carrier_cos + analyticR * carrier_sin)
    return isb_usb + isb_lsb

def ejecutar_modulacion(tipo_modulacion, banda):
    try:
        if banda == "LSB":
            fs, audio = cargar_audio(archivo_baja)
        elif banda == "USB":
            fs, audio = cargar_audio(archivo_alta)
        else:
            estado_var.set("❌ Banda no reconocida")
            return

        estado_var.set(f"⚙️ Modulando {tipo_modulacion}-{banda}...")
        root.update()

        graficar_senal_tiempo_frecuencia(audio, fs, "Audio Original", usar_analitica=True)

        tono_i = generar_tono(TONO_INICIO, DUR_TONO, FS)
        tono_f = generar_tono(TONO_FIN, DUR_TONO, FS)

        if tipo_modulacion == "SC":
            salida = modulacion_ssb(audio, banda)
        elif tipo_modulacion == "FC":
            salida = modulacion_ssb_fc(audio, banda)
        else:
            estado_var.set("❌ Tipo no reconocido")
            return

        graficar_senal_tiempo_frecuencia(salida, fs, f"Modulada {tipo_modulacion}-{banda}", usar_analitica=True)

        total = np.concatenate((tono_i, salida, tono_f))
        estado_var.set(f"🔊 Reproduciendo {tipo_modulacion}-{banda}")
        reproducir_senal(total, FS)
        estado_var.set(f"✅ {tipo_modulacion}-{banda} completado.")
    except Exception as e:
        estado_var.set(f"❌ Error: {e}")
        print("[ERROR]", e)

def ejecutar_isb():
    try:
        fsL, audioL = cargar_audio(archivo_L_ISB)
        fsR, audioR = cargar_audio(archivo_R_ISB)

        # Asegurar que ambas señales tengan la misma longitud
        min_len = min(len(audioL), len(audioR))
        audioL = audioL[:min_len]
        audioR = audioR[:min_len]



        estado_var.set("⚙️ Modulando ISB...")
        root.update()

        graficar_senal_tiempo_frecuencia(audioL, fsL, "Audio L", usar_analitica=True)
        graficar_senal_tiempo_frecuencia(audioR, fsR, "Audio R", usar_analitica=True)

        tono_i = generar_tono(TONO_INICIO, DUR_TONO, FS)
        tono_f = generar_tono(TONO_FIN, DUR_TONO, FS)
        isb = modulacion_isb(audioL, audioR)

        graficar_senal_tiempo_frecuencia(isb, FS, "Modulada ISB", usar_analitica=True)

        total = np.concatenate((tono_i, isb, tono_f))
        estado_var.set("🔊 Reproduciendo ISB")
        reproducir_senal(total, FS)
        estado_var.set("✅ ISB completado.")
    except Exception as e:
        estado_var.set(f"❌ Error ISB: {e}")
        print("[ERROR]", e)

# === Interfaz Gráfica ===
imagen_path = "walki.png"
imagen = Image.open(imagen_path)
ancho, alto = imagen.size

botones = {
    "GRABAR BAJA": (105, 163, 50, 16),
    "GRABAR ALTA": (165, 163, 50, 16),
    "REPRODUCIR BAJA": (250, 163, 50, 16),
    "REPRODUCIR ALTA": (305, 163, 50, 16),
    "SSB-SCL": (77, 216, 71, 15),
    "SSB-SCU": (177, 216, 69, 15),
    "SSB-FCL": (272, 216, 70, 15),
    "SSB-FCU": (78, 262, 69, 15),
    "ISB": (177, 262, 70, 15),
    "ESC": (313, 384, 26, 13)
}

root = tk.Tk()
root.title("Modulador AM")
root.geometry(f"{ancho}x{alto}")
root.resizable(False, False)

imagen_tk = ImageTk.PhotoImage(imagen)
canvas = tk.Canvas(root, width=ancho, height=alto)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=imagen_tk)

estado_var = tk.StringVar(value="")
tk.Label(root, textvariable=estado_var, bg="white", fg="black", font=("Arial", 14)).place(x=70, y=315, width=400)

for nombre, (x, y, w, h) in botones.items():
    if nombre == "GRABAR BAJA":
        comando = lambda: grabar_audio(archivo_baja)
    elif nombre == "GRABAR ALTA":
        comando = lambda: grabar_audio(archivo_alta)
    elif nombre == "REPRODUCIR BAJA":
        comando = lambda: reproducir_audio(archivo_baja)
    elif nombre == "REPRODUCIR ALTA":
        comando = lambda: reproducir_audio(archivo_alta)
    elif nombre == "ESC":
        comando = root.destroy
    elif nombre == "SSB-SCL":
        comando = lambda: ejecutar_modulacion("SC", "LSB")
    elif nombre == "SSB-SCU":
        comando = lambda: ejecutar_modulacion("SC", "USB")
    elif nombre == "SSB-FCU":
        comando = lambda: ejecutar_modulacion("FC", "USB")
    elif nombre == "SSB-FCL":
        comando = lambda: ejecutar_modulacion("FC", "LSB")
    elif nombre == "ISB":
        comando = ejecutar_isb
    else:
        comando = lambda n=nombre: estado_var.set(f"Presionado: {n}")

    tk.Button(root, text=nombre, command=comando,
              bg="#222", fg="white", font=("Arial", 9)).place(x=x, y=y, width=w, height=h)

root.mainloop()

