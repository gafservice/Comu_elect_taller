import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, resample
import sounddevice as sd  # Reemplazamos IPython por sounddevice

# Funciones auxiliares

def cargar_audio(archivo, fs_nueva, duracion):
    fs_orig, audio_orig = wavfile.read(archivo)
    audio_orig = audio_orig.astype(np.float32)
    N = int(fs_nueva * duracion)
    audio_resample = resample(audio_orig, N)
    audio_resample /= np.max(np.abs(audio_resample))  # Normaliza el audio
    return audio_resample, fs_nueva

def generar_tono(frecuencia, duracion, fs):
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)  # Tiempo de la señal
    tono = 8191 * np.sin(2 * np.pi * frecuencia * t)  # Tono con amplitud reducida
    return tono.astype(np.int16)

def modulación_ssb(audio, t, fc):
    carrier_cos = np.cos(2*np.pi*fc*t)
    carrier_sin = np.sin(2*np.pi*fc*t)
    analytic = np.imag(hilbert(audio))
    
    ssb_sc_lsb = np.real(audio * carrier_cos + analytic * carrier_sin)  # LSB
    ssb_sc_usb = np.real(audio * carrier_cos - analytic * carrier_sin)  # USB
    
    return ssb_sc_lsb, ssb_sc_usb

def modulación_isb(audio, audio2, t, fc):
    carrier_cos = np.cos(2*np.pi*fc*t)
    carrier_sin = np.sin(2*np.pi*fc*t)
    analytic = np.imag(hilbert(audio))
    analytic2 = np.imag(hilbert(audio2))
    
    isb_usb = np.real(audio * carrier_cos - analytic * carrier_sin)
    isb_lsb = np.real(audio2 * carrier_cos + analytic2 * carrier_sin)
    
    return isb_usb, isb_lsb

def graficar_audio(t, audio, f, audio_fft_db, title):
    # Cambiar el tamaño de la figura
    plt.figure(figsize=(5, 4))  # Cambié el tamaño a 1/4 del tamaño original

    plt.subplot(2, 1, 1)
    plt.plot(t, audio)
    plt.title(f"Gráfica del {title}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f[:len(f)//2], audio_fft_db[:len(f)//2])
    plt.title(f"Espectro del {title}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.grid()

    plt.tight_layout()
    plt.show()


def menu():
    print("\nSeleccione el tipo de modulación:")
    print("1. Modulación SSB (Single Sideband)")
    print("2. Modulación ISB (Independent Sideband)")
    print("3. Salir")

    opcion = input("Ingrese el número de su selección: ")

    if opcion == '1':
        return "SSB"
    elif opcion == '2':
        return "ISB"
    elif opcion == '3':
        return "Salir"
    else:
        print("Opción no válida. Inténtelo de nuevo.")
        return menu()


# Menú principal para seleccionar el tipo de modulación
def ejecutar_modulacion():
    # Parámetros del sistema
    fs = 120000
    duracion = 5
    fc = 10000  # Frecuencia portadora

    # Cargar los audios desde los archivos subidos
    audio, fs = cargar_audio("audio_mono.wav", fs, duracion)  # Usamos audio mono
    audio2, fs2 = cargar_audio("audio_L_ISB.wav", fs, duracion)

    t = np.arange(len(audio)) / fs  # Vector de tiempo para la señal remuestreada
    f = np.fft.fftfreq(len(audio), 1/fs)

    # Selección de modulación
    opcion = menu()

    if opcion == "SSB":
        # Realizamos modulación SSB
        ssb_sc_lsb, ssb_sc_usb = modulación_ssb(audio, t, fc)
        
        # Realizamos la FFT de las señales SSB
        ssb_sc_lsb_fft = np.abs(np.fft.fft(ssb_sc_lsb))
        ssb_sc_usb_fft = np.abs(np.fft.fft(ssb_sc_usb))

        # Conversión a dB
        ssb_sc_lsb_fft_db = 20 * np.log10(ssb_sc_lsb_fft)
        ssb_sc_usb_fft_db = 20 * np.log10(ssb_sc_usb_fft)

        # Graficamos las señales SSB
        graficar_audio(t, ssb_sc_lsb, f, ssb_sc_lsb_fft_db, "Modulación SSB-SC-LSB")
        graficar_audio(t, ssb_sc_usb, f, ssb_sc_usb_fft_db, "Modulación SSB-SC-USB")
        
        # Reproducir el audio modulado (SSB-USB)
        sd.play(ssb_sc_usb, fs)
        sd.wait()
    
    elif opcion == "ISB":
        # Cargar el segundo archivo para ISB
        audio2, fs2 = cargar_audio("audio_R_ISB.wav", fs, duracion)

        # Realizamos modulación ISB
        isb_usb, isb_lsb = modulación_isb(audio, audio2, t, fc)
        
        # Realizamos la FFT de las señales ISB
        isb_fft_usb = np.abs(np.fft.fft(isb_usb))
        isb_fft_lsb = np.abs(np.fft.fft(isb_lsb))

        # Conversión a dB
        isb_fft_db_usb = 20 * np.log10(isb_fft_usb)
        isb_fft_db_lsb = 20 * np.log10(isb_fft_lsb)

        # Graficamos las señales ISB
        graficar_audio(t, isb_usb, f, isb_fft_db_usb, "Modulación ISB-USB")
        graficar_audio(t, isb_lsb, f, isb_fft_db_lsb, "Modulación ISB-LSB")
        
        # Reproducir el audio modulado (ISB-USB)
        sd.play(isb_usb, fs)
        sd.wait()

    elif opcion == "Salir":
        print("Saliendo del programa.")
        return


# Llamar a la función para ejecutar el menú y procesar la modulación
ejecutar_modulacion()

