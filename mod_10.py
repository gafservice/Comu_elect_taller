import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert
import sounddevice as sd

# === Parámetros globales ===
FS = 44100
FC = 10000  # Frecuencia de la portadora
DUR_TONO = 0.2  # segundos
TONO_INICIO = 7000
TONO_FIN = 5000

# === Funciones auxiliares ===

def cargar_audio(nombre_archivo):
    fs, audio = wavfile.read(nombre_archivo)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio /= np.max(np.abs(audio))
    return fs, audio

def generar_tono(freq, duracion, fs):
    t = np.arange(int(fs * duracion)) / fs
    tono = 0.7 * np.sin(2 * np.pi * freq * t)
    return tono

def reproducir_senal(senal, fs):
    sd.play(senal, fs, blocking=True)

def graficar_senal_tiempo(t, s, titulo, ax):
    ax.plot(t[:int(FS*0.005)], s[:int(FS*0.005)])
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    ax.grid()

def graficar_fft(s, titulo, ax):
    N = len(s)
    f = np.fft.fftfreq(N, d=1/FS)
    S_mag = np.abs(np.fft.fft(s))
    ax.plot(f[:N//2], S_mag[:N//2])
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Magnitud")
    ax.grid()

# === Modulaciones ===

def modulacion_ssb(audio, tipo):
    t = np.arange(len(audio)) / FS
    analytic = hilbert(audio)
    if tipo == "USB":
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * FC * t))
    else:
        ssb = np.real(analytic * np.exp(-1j * 2 * np.pi * FC * t))
    return ssb

def modulacion_ssb_fc(audio, tipo):
    t = np.arange(len(audio)) / FS
    carrier = np.cos(2 * np.pi * FC * t)
    ssb_sc = modulacion_ssb(audio, tipo)
    return ssb_sc + audio * carrier

def modulacion_isb(audio_L, audio_R):
    t = np.arange(len(audio_L)) / FS
    analytic_L = hilbert(audio_L)
    analytic_R = hilbert(audio_R)
    lsb = np.real(analytic_L * np.exp(-1j * 2 * np.pi * FC * t))
    usb = np.real(analytic_R * np.exp(1j * 2 * np.pi * FC * t))
    isb = lsb + usb
    return isb, lsb, usb

# === Menú de selección ===

def menu():
    while True:
        print("\nSeleccione el tipo de modulación:")
        print("1. Modulación SSB-SC (Single Sideband - Suppressed Carrier)")
        print("2. Modulación SSB-FC (Single Sideband - Full Carrier)")
        print("3. Modulación ISB (Independent Sideband)")
        print("4. Salir")
        opcion = input("Ingrese el número de su selección: ")

        if opcion == "1":
            tipo = input("Seleccione banda lateral (1 para USB, 2 para LSB): ")
            tipo = "USB" if tipo == "1" else "LSB"
            fs, audio = cargar_audio("audio_mono.wav")
            ssb = modulacion_ssb(audio, tipo)
            tono_i = generar_tono(TONO_INICIO, DUR_TONO, FS)
            tono_f = generar_tono(TONO_FIN, DUR_TONO, FS)
            total = np.concatenate((tono_i, ssb, tono_f))
            reproducir_senal(total, FS)
            mostrar_graficas(audio, ssb, "SSB-SC " + tipo)

        elif opcion == "2":
            tipo = input("Seleccione banda lateral (1 para USB, 2 para LSB): ")
            tipo = "USB" if tipo == "1" else "LSB"
            fs, audio = cargar_audio("audio_mono.wav")
            ssb_fc = modulacion_ssb_fc(audio, tipo)
            tono_i = generar_tono(TONO_INICIO, DUR_TONO, FS)
            tono_f = generar_tono(TONO_FIN, DUR_TONO, FS)
            total = np.concatenate((tono_i, ssb_fc, tono_f))
            reproducir_senal(total, FS)
            mostrar_graficas(audio, ssb_fc, "SSB-FC " + tipo)

        elif opcion == "3":
            fsL, audioL = cargar_audio("audio_L_ISB.wav")
            fsR, audioR = cargar_audio("audio_R_ISB.wav")
            isb, lsb, usb = modulacion_isb(audioL, audioR)
            tono_i = generar_tono(TONO_INICIO, DUR_TONO, FS)
            tono_f = generar_tono(TONO_FIN, DUR_TONO, FS)
            total = np.concatenate((tono_i, isb, tono_f))
            reproducir_senal(total, FS)
            mostrar_grafica_isb(audioL, audioR, lsb, usb)

        elif opcion == "4":
            print("Programa finalizado.")
            break
        else:
            print("❌ Opción no válida. Intente de nuevo.")

def mostrar_graficas(audio, modulado, titulo):
    t = np.arange(len(audio)) / FS
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    graficar_senal_tiempo(t, audio, "Audio original", axs[0])
    graficar_senal_tiempo(t, modulado, "Señal " + titulo, axs[1])
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    graficar_fft(audio, "Espectro del audio original", axs[0])
    graficar_fft(modulado, "Espectro de la señal " + titulo, axs[1])
    plt.tight_layout()
    plt.show()

def mostrar_grafica_isb(audioL, audioR, lsb, usb):
    t = np.arange(len(audioL)) / FS
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(t[:int(FS*0.005)], audioL[:int(FS*0.005)], label="Canal L (LSB)")
    axs[0].plot(t[:int(FS*0.005)], audioR[:int(FS*0.005)], label="Canal R (USB)")
    axs[0].set_title("Audio izquierdo y derecho")
    axs[0].legend()
    axs[0].grid()

    f = np.fft.fftfreq(len(lsb), d=1/FS)
    S_lsb = np.abs(np.fft.fft(lsb))
    S_usb = np.abs(np.fft.fft(usb))
    axs[1].plot(f[:len(f)//2], S_lsb[:len(f)//2], label="LSB", linestyle='--')
    axs[1].plot(f[:len(f)//2], S_usb[:len(f)//2], label="USB", linestyle='-')
    axs[1].set_title("Espectro combinado LSB/USB")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.show()

# === Ejecutar ===
menu()

