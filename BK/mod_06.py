import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def cargar_audio(ruta):
    fs, audio = wavfile.read(ruta)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio /= np.max(np.abs(audio))
    return fs, audio

def agregar_tonos(señal, fs):
    t_tono = np.arange(0, int(fs * 0.2)) / fs
    tono_inicio = 0.7 * np.sin(2 * np.pi * 7000 * t_tono)  # Tono de inicio: 7000 Hz
    tono_fin = 0.7 * np.sin(2 * np.pi * 5000 * t_tono)     # Tono de fin: 5000 Hz
    return np.concatenate((tono_inicio, señal, tono_fin))

def graficar_senal(t, señal, fs, titulo):
    plt.figure()
    plt.plot(t[:int(fs * 0.005)], señal[:int(fs * 0.005)])
    plt.title(f"{titulo} (tiempo, primeros 5 ms)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid()

def graficar_espectro(señal, fs, titulo):
    N = len(señal)
    f = np.fft.fftfreq(N, d=1/fs)
    S = np.abs(np.fft.fft(señal))
    plt.figure()
    plt.plot(f[:N//2], S[:N//2])
    plt.title(f"{titulo} (espectro)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid()

def ssb_mod(audio, fs, fc, tipo='SC', banda='USB'):
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    portadora = np.exp(1j * 2 * np.pi * fc * t)
    ssb = analytic * portadora

    if banda == 'LSB':
        ssb = np.real(np.conj(ssb))
    else:
        ssb = np.real(ssb)

    if tipo == 'FC':
        ssb += np.cos(2 * np.pi * fc * t)

    return ssb, t

def isb_mod(audioL, audioR, fs, fc):
    t = np.arange(len(audioL)) / fs
    analyticL = hilbert(audioL)
    analyticR = hilbert(audioR)
    usb = np.real(analyticL * np.exp(1j * 2 * np.pi * fc * t))
    lsb = np.real(np.conj(analyticR * np.exp(1j * 2 * np.pi * fc * t)))
    return usb + lsb, t

def menu_modulacion():
    while True:
        fc = 10000  # Hz fijo
        fs, audio = cargar_audio("audio_mono.wav")

        print("\n=== MENÚ DE MODULACIÓN ===")
        print("1. SSB-SC (Portadora suprimida)")
        print("2. SSB-FC (Con portadora)")
        print("3. ISB (Independiente con audio_L_ISB y audio_R_ISB)")
        print("4. Salir")
        opcion = input("Ingrese 1, 2, 3 o 4: ")

        if opcion in ['1', '2']:
            tipo = 'SC' if opcion == '1' else 'FC'

            print("Seleccione banda lateral:")
            print("1. USB (Upper Side Band)")
            print("2. LSB (Lower Side Band)")
            banda_opcion = input("Ingrese 1 o 2: ").strip()

            if banda_opcion == '1':
                banda = 'USB'
            elif banda_opcion == '2':
                banda = 'LSB'
            else:
                print("❌ Selección inválida.")
                continue

            señal_modulada, t = ssb_mod(audio, fs, fc, tipo, banda)

        elif opcion == '3':
            fs_L, audioL = cargar_audio("audio_L_ISB.wav")
            fs_R, audioR = cargar_audio("audio_R_ISB.wav")
            if fs_L != fs_R:
                print("❌ Las frecuencias de muestreo no coinciden entre L y R.")
                continue
            señal_modulada, t = isb_mod(audioL, audioR, fs_L, fc)
            fs = fs_L

        elif opcion == '4':
            print("Saliendo del programa...")
            break

        else:
            print("❌ Opción inválida.")
            continue

        señal_total = agregar_tonos(señal_modulada, fs)

        print("🔊 Reproduciendo señal modulada con tonos...")
        sd.play(señal_total, fs, blocking=True)
        sd.wait()

        wavfile.write("salida_modulada.wav", fs, (señal_total * 32767).astype(np.int16))
        print("✅ Señal guardada como salida_modulada.wav")

        graficar_senal(t, señal_modulada, fs, "Señal Modulada")
        graficar_espectro(señal_modulada, fs, "Espectro de la Señal Modulada")
        plt.show()

# Ejecutar el menú principal
if __name__ == "__main__":
    menu_modulacion()

