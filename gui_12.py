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
archivo_estereo = 'audio_estereo.wav'
archivo_mono = 'audio_mono.wav'
archivo_L_ISB = 'audio_L_ISB.wav'
archivo_R_ISB = 'audio_R_ISB.wav'

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

def actualizar_tiempo(estado_var, grabando_flag, duracion):
    def actualizar(segundos=[0]):
        if not grabando_flag[0]:
            return
        estado_var.set(f"🎧 Grabando... {segundos[0]}s")
        root.update()
        segundos[0] += 1
        if segundos[0] <= duracion:
            root.after(1000, lambda: actualizar(segundos))
    actualizar()

def grabar_audio():
    thread = threading.Thread(target=grabar_en_hilo)
    thread.start()

def grabar_en_hilo():
    estado_var.set("🎧 Preparando grabación...")
    root.update()
    try:
        dispositivo = sd.default.device[0]
        info = sd.query_devices(dispositivo)

        if info['max_input_channels'] < 2:
            raise RuntimeError("El dispositivo no tiene al menos 2 canales de entrada.")

        grabando_flag = [True]
        root.after(0, lambda: actualizar_tiempo(estado_var, grabando_flag, duracion))

        audio_estereo = sd.rec(int(duracion * fs), samplerate=fs, channels=2, dtype='int16', device=dispositivo)
        sd.wait()
        grabando_flag[0] = False

        estado_var.set("📃 Guardando estéreo...")
        root.update()
        write(archivo_estereo, fs, audio_estereo)

        estado_var.set("🎚 Procesando mono...")
        root.update()
        audio_mono = np.mean(audio_estereo, axis=1).astype(np.int16)
        write(archivo_mono, fs, suavizar(audio_mono))

        write(archivo_L_ISB, fs, audio_estereo[:, 0])
        write(archivo_R_ISB, fs, audio_estereo[:, 1])

        estado_var.set("✅ Grabación y archivos Listos.")
        root.update()

    except Exception as e:
        estado_var.set(f"❌ Error: {e}")
        print("[ERROR]", e)

def reproducir_todo():
    try:
        secuencias = [
            (archivo_estereo, "🔊 Estéreo"),
            (archivo_mono, "🔊 Mono suavizado"),
            (archivo_L_ISB, "🔊 ISB L"),
            (archivo_R_ISB, "🔊 ISB R")
        ]
        for archivo, descripcion in secuencias:
            if not os.path.exists(archivo):
                estado_var.set(f"⚠️ Falta archivo: {archivo}")
                root.update()
                continue
            estado_var.set(descripcion)
            root.update()
            fs_leido, datos = read(archivo)
            sd.play(datos, fs_leido)
            sd.wait()
        estado_var.set("✅ Reproducción completada.")
    except Exception as e:
        estado_var.set(f"❌ Error reproducción: {e}")
        print("[ERROR]", e)

def ejecutar_modulacion(tipo_modulacion, banda):
    try:
        fs, audio = cargar_audio(archivo_mono)
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
    "GRABAR":     (113, 163, 72, 14),
    "REPRODUCIR": (252, 163, 93, 14),
    "SSB-SCL":    (77, 216, 71, 15),
    "SSB-SCU":    (177, 216, 69, 15),
    "SSB-FCL":    (272, 216, 70, 15),
    "SSB-FCU":    (78, 262, 69, 15),
    "ISB":        (177, 262, 70, 15),
    "ESC":        (313, 384, 26, 13)
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
tk.Label(root, textvariable=estado_var, bg="white", fg="black", font=("Arial", 14)).place(x=70, y=315, width=260)

for nombre, (x, y, w, h) in botones.items():
    if nombre == "GRABAR":
        comando = grabar_audio
    elif nombre == "REPRODUCIR":
        comando = reproducir_todo
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

