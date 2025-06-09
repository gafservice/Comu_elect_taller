import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import os

fs = 44100
grabacion_actual = None

def grabar_audio(status_label):
    global grabacion_actual
    status_label.config(text="Grabando...")
    root.update_idletasks()
    grabacion_actual = sd.rec(int(5 * fs), samplerate=fs, channels=1)
    sd.wait()
    write("grabacion.wav", fs, (grabacion_actual * 32767).astype(np.int16))
    status_label.config(text="Grabación finalizada.")

def reproducir_audio(status_label):
    global grabacion_actual
    if grabacion_actual is None and os.path.exists("grabacion.wav"):
        _, grabacion_actual = read("grabacion.wav")
        grabacion_actual = grabacion_actual.astype(np.float32)
        grabacion_actual /= np.max(np.abs(grabacion_actual))
    if grabacion_actual is not None:
        status_label.config(text="Reproduciendo...")
        sd.play(grabacion_actual, fs)
        sd.wait()
        status_label.config(text="Listo")
    else:
        messagebox.showwarning("Advertencia", "No hay grabación disponible.")

def iniciar_gui():
    global root
    root = tk.Tk()
    root.title("Modulador AM Avanzado")
    root.geometry("512x768")
    root.resizable(False, False)

    fondo_img = Image.open("walki.png")
    fondo_tk = ImageTk.PhotoImage(fondo_img)
    tk.Label(root, image=fondo_tk).place(x=0, y=0)

    def boton(x, y, texto, comando, w=90, h=35):
        tk.Button(root, text=texto, command=comando, bg="gray", fg="white",
                  font=("Arial", 9, "bold"), relief="flat", cursor="hand2").place(x=x, y=y, width=w, height=h)

    def entrada(x, y, w=140, h=30):
        e = tk.Entry(root, font=("Arial", 9), justify="center")
        e.place(x=x, y=y, width=w, height=h)
        return e

    # Etiqueta para mostrar el estado del sistema
    status_label = tk.Label(root, text="Listo", font=("Arial", 10), fg="blue", bg="white", relief="sunken", anchor="center")
    status_label.place(x=156, y=600, width=200, height=30)

    boton(125, 205, "Grabar", lambda: grabar_audio(status_label), 110, 40)
    boton(300, 205, "Reproducir", lambda: reproducir_audio(status_label), 120, 40)

    # Botones de tipo de modulación (solo efectos visuales en este ejemplo)
    boton(85, 287, "DSB", lambda: messagebox.showinfo("Modulación", "DSB"))
    boton(210, 287, "CSB", lambda: messagebox.showinfo("Modulación", "CSB"))
    boton(330, 284, "SSB", lambda: messagebox.showinfo("Modulación", "SSB"))

    boton(85, 348, "SSB-FC", lambda: messagebox.showinfo("Modulación", "SSB-FC"))
    boton(210, 348, "SSB-SC", lambda: messagebox.showinfo("Modulación", "SSB-SC"))
    boton(330, 348, "ISB", lambda: messagebox.showinfo("Modulación", "ISB"))

    entrada(70, 443)     # Error de fase
    entrada(265, 443)    # Error de frecuencia

    banda = tk.StringVar(value="USB")
    tk.Radiobutton(root, text="USB", variable=banda, value="USB", font=("Arial", 16), bg="white").place(x=138, y=496)
    tk.Radiobutton(root, text="LSB", variable=banda, value="LSB", font=("Arial", 16), bg="white").place(x=276, y=496)

    boton(388, 527, "ESC", root.destroy, 60, 30)

    root.mainloop()

if __name__ == "__main__":
    iniciar_gui()

