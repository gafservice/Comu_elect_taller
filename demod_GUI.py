import threading
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threads
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import time
import os
from tkinter import Tk, Button, Label, Text, Scrollbar, END, DISABLED, NORMAL

def butter_lowpass(cutoff, fs, order=6):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return b, a

def butter_bandpass(cutoff, lowcut, highcut, fs, order):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
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
    return f"{base}{i:03d}.wav"

class AudioListenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Escucha y Demodulación AM-SSB")
        self.fs = 44100
        self.fc = 10000
        self.blocksize = int(0.1 * self.fs)
        self.dur_max_mensaje = 10
        self.umbral_inicio = 10
        self.umbral_fin = 5
        self.listening = False
        self.inicio_detectado = False
        self.espectro_guardado = False
        self.mensaje = []
        self.thread = None
        self.stream = None

        # UI Elements
        self.start_button = Button(root, text="▶️ Iniciar escucha", command=self.start_listening)
        self.start_button.pack(pady=5)
        self.stop_button = Button(root, text="⏹️ Detener escucha", command=self.stop_listening, state=DISABLED)
        self.stop_button.pack(pady=5)
        self.status_label = Label(root, text="Estado: Inactivo")
        self.status_label.pack(pady=5)

        self.log = Text(root, height=20, width=70, state=DISABLED)
        self.log.pack(padx=10, pady=5)
        scrollbar = Scrollbar(root, command=self.log.yview)
        scrollbar.pack(side='right', fill='y')
        self.log.config(yscrollcommand=scrollbar.set)

    def log_message(self, msg):
        self.log.config(state=NORMAL)
        self.log.insert(END, msg + "\n")
        self.log.see(END)
        self.log.config(state=DISABLED)

    def start_listening(self):
        if not self.listening:
            self.listening = True
            self.inicio_detectado = False
            self.espectro_guardado = False
            self.mensaje = []
            self.start_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
            self.status_label.config(text="Estado: Escuchando...")
            self.log_message("🔁 Sistema activo. Esperando tono de 13000 Hz...")
            self.thread = threading.Thread(target=self.listen_loop, daemon=True)
            self.thread.start()

    def stop_listening(self):
        if self.listening:
            self.listening = False
            self.status_label.config(text="Estado: Detenido")
            self.start_button.config(state=NORMAL)
            self.stop_button.config(state=DISABLED)
            self.log_message("🛑 Escucha detenida.")
            if self.stream:
                self.stream.close()

    def listen_loop(self):
        while self.listening:
            self.log_message("🕑 Esperando 0.5 segundos antes de iniciar...")
            time.sleep(0.5)
            self.mensaje = []
            self.inicio_detectado = False
            self.espectro_guardado = False

            def callback_inicial(indata, frames, time_info, status):
                if not self.listening:
                    raise sd.CallbackStop()
                bloque = indata[:, 0]
                detectado, f, S = detectar_tono(bloque, 15000, self.fs, margen=30, umbral=self.umbral_inicio)
                if detectado and not self.inicio_detectado:
                    self.inicio_detectado = True
                    self.mensaje.append(bloque.copy())
                    self.log_message("✅ Tono de inicio detectado.")
                    if not self.espectro_guardado:
                        nombre_png = siguiente_nombre().replace('.wav', '_espectro.png')
                        plt.figure()
                        plt.plot(f, S)
                        plt.title("Espectro al detectar el tono de inicio (15000 Hz)")
                        plt.xlabel("Frecuencia [Hz]")
                        plt.ylabel("Magnitud")
                        plt.grid()
                        plt.savefig(nombre_png)
                        plt.close()
                        self.espectro_guardado = True
                        self.log_message(f"🖼️ Espectro guardado como '{nombre_png}'")
                elif self.inicio_detectado:
                    self.mensaje.append(bloque.copy())

            with sd.InputStream(callback=callback_inicial, blocksize=self.blocksize, channels=1, samplerate=self.fs) as self.stream:
                while not self.inicio_detectado and self.listening:
                    time.sleep(0.05)
                if not self.listening:
                    break

            if not self.listening:
                break

            self.log_message("⏺️ Grabando mensaje hasta detectar tono de fin (3000 Hz)...")

            def callback_mensaje(indata, frames, time_info, status):
                if not self.listening:
                    raise sd.CallbackStop()
                bloque = indata[:, 0]
                detectado_fin, _, _ = detectar_tono(bloque, 3000, self.fs, margen=30, umbral=self.umbral_fin)
                if not detectado_fin:
                    self.mensaje.append(bloque.copy())
                else:
                    self.log_message("✅ Tono de fin detectado.")
                    raise sd.CallbackStop()

            with sd.InputStream(callback=callback_mensaje, blocksize=self.blocksize, channels=1, samplerate=self.fs) as self.stream:
                try:
                    sd.sleep(int(self.dur_max_mensaje * 1000))
                except sd.CallbackStop:
                    pass
                if not self.listening:
                    break

            if not self.mensaje:
                self.log_message("⚠️ No se grabó ningún mensaje.")
                continue

            self.log_message("🎞 Procesando señal...")

            mensaje_array = np.concatenate(self.mensaje)
            b_bp, a_bp = butter_bandpass(self.fc, 9000, 11000, self.fs,8)
            mensaje_filtrado = filtfilt(b_bp, a_bp, mensaje_array)
            t = np.arange(len(mensaje_array)) / self.fs
            portadora = np.cos(2 * np.pi * self.fc * t)
            baseband = mensaje_filtrado * portadora
            b, a = butter_lowpass(4000, self.fs)
            audio = filtfilt(b, a, baseband)
            audio /= np.max(np.abs(audio))

            plt.figure()
            plt.plot(t, audio)
            plt.title("Mensaje demodulado (dominio del tiempo)")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Amplitud")
            plt.grid()
            nombre_senal = siguiente_nombre().replace('.wav', '_tiempo.png')
            plt.savefig(nombre_senal)
            plt.close()
            self.log_message(f"🖼️ Señal en el tiempo guardada como '{nombre_senal}'")

            output_file = nombre_senal.replace('_tiempo.png', '.wav')
            self.log_message(f"🔊 Reproduciendo mensaje demodulado...")
            sd.play(audio, self.fs)
            sd.wait()
            write(output_file, self.fs, (audio * 32767).astype(np.int16))
            self.log_message(f"💾 Audio guardado como '{output_file}'\n")
            self.log_message("🔁 Reiniciando escucha...\n")

        # When loop ends
        self.listening = False
        self.status_label.config(text="Estado: Inactivo")
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)

if __name__ == '__main__':
    root = Tk()
    app = AudioListenerApp(root)
    root.mainloop()
