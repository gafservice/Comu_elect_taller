import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import os
import time

def butter_lowpass(cutoff, fs, order=6):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return b, a

def detectar_tono(bloque, tono, fs, margen=30, umbral=10):
    N = len(bloque)
    f = np.fft.rfftfreq(N, 1/fs)
    S = np.abs(np.fft.rfft(bloque * np.hanning(N)))
    idx = np.where((f >= tono - margen) & (f <= tono + margen))[0]
    if len(idx) == 0:
        return False, f, S
    energia_tono = np.max(S[idx])
    energia_promedio = np.mean(S)
    return energia_tono > energia_promedio * umbral, f, S

def siguiente_nombre():
    base = "grabacion_"
    i = 1
    while os.path.exists(f"{base}{i:03d}.wav"):
        i += 1
    return f"{base}{i:03d}"

def iniciar_proceso_con_acumulador(gui):
    fs = 44100
    fc = 10000
    blocksize = int(0.1 * fs)
    dur_max_mensaje = 10
    umbral_inicio = 10
    umbral_fin = 5
    acumulador = []

    sd.default.samplerate = fs
    sd.default.channels = 1

    gui.estado.set("🟡 Escuchando tono de inicio (7000 Hz)...")
    time.sleep(0.5)

    mensaje = []
    inicio_detectado = False
    espectro_guardado = False

    def callback_inicial(indata, frames, time_info, status):
        nonlocal inicio_detectado, espectro_guardado, mensaje, acumulador
        bloque = indata[:, 0]
        acumulador.append(bloque.copy())

        if not espectro_guardado and len(acumulador) * blocksize >= fs:
            bloque_largo = np.concatenate(acumulador)
            detectado, f, S = detectar_tono(bloque_largo, 7000, fs, margen=30, umbral=umbral_inicio)
            if detectado:
                inicio_detectado = True
                mensaje.append(bloque_largo.copy())
                gui.estado.set("✅ Tono de inicio detectado. Grabando...")

                fig = plt.Figure(figsize=(5, 2), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(f, S)
                ax.set_title("Espectro acumulado al detectar tono")
                ax.set_xlabel("Frecuencia [Hz]")
                ax.set_ylabel("Magnitud")
                ax.grid(True)
                gui.mostrar_grafica(fig, 0)
                espectro_guardado = True
        elif inicio_detectado:
            mensaje.append(bloque.copy())

    with sd.InputStream(callback=callback_inicial, blocksize=blocksize):
        while not inicio_detectado:
            time.sleep(0.05)

    def callback_mensaje(indata, frames, time_info, status):
        nonlocal mensaje
        bloque = indata[:, 0]
        detectado_fin, _, _ = detectar_tono(bloque, 5000, fs, margen=30, umbral=umbral_fin)
        if not detectado_fin:
            mensaje.append(bloque.copy())
        else:
            gui.estado.set("✅ Tono de fin detectado. Procesando...")
            raise sd.CallbackStop()

    with sd.InputStream(callback=callback_mensaje, blocksize=blocksize):
        try:
            sd.sleep(int(dur_max_mensaje * 1000))
        except sd.CallbackStop:
            pass

    mensaje = np.concatenate(mensaje)
    t = np.arange(len(mensaje)) / fs
    portadora = np.cos(2 * np.pi * fc * t)
    baseband = mensaje * portadora
    b, a = butter_lowpass(4000, fs)
    audio = filtfilt(b, a, baseband)
    audio /= np.max(np.abs(audio))

    fig2 = plt.Figure(figsize=(5, 2), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.plot(t, audio)
    ax2.set_title("Señal demodulada (tiempo)")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True)
    gui.mostrar_grafica(fig2, 1)

    nombre_base = siguiente_nombre()
    wavname = nombre_base + ".wav"
    write(wavname, fs, (audio * 32767).astype(np.int16))
    gui.estado.set(f"🎧 Reproduciendo y guardado como {wavname}")
    sd.play(audio, fs)
    sd.wait()
    gui.estado.set("🔁 Proceso completado. Listo para reiniciar.")

class DemodGUI:
    def __init__(self, master):
        self.master = master
        master.title("Demodulador AM - GUI Autónoma")

        self.estado = tk.StringVar()
        self.estado.set("⏳ Esperando inicio...")

        ttk.Label(master, textvariable=self.estado, font=("Arial", 12)).pack(pady=10)
        ttk.Button(master, text="▶ Iniciar Demodulación", command=self.iniciar).pack(pady=10)

        self.frames = [
            ttk.LabelFrame(master, text="Espectro acumulado"),
            ttk.LabelFrame(master, text="Señal Demodulada")
        ]
        for frame in self.frames:
            frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = [None, None]

    def iniciar(self):
        self.limpiar_graficas()
        threading.Thread(target=iniciar_proceso_con_acumulador, args=(self,), daemon=True).start()

    def mostrar_grafica(self, fig, idx):
        if self.canvas[idx]:
            self.canvas[idx].get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.frames[idx])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas[idx] = canvas

    def limpiar_graficas(self):
        for idx in range(2):
            if self.canvas[idx]:
                self.canvas[idx].get_tk_widget().destroy()
                self.canvas[idx] = None

if __name__ == '__main__':
    
    root = tk.Tk()
    root.geometry("1000x700")  # ancho x alto en píxeles

    gui = DemodGUI(root)
    root.mainloop()

