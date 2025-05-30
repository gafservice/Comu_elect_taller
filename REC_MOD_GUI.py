import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

fc = 10000  # Frecuencia de la portadora fija

def tono(frec, dur, fs):
    t = np.arange(0, int(fs * dur)) / fs
    return 0.7 * np.sin(2 * np.pi * frec * t)

def graficar_espectro(x, fs, titulo):
    N = len(x)
    f = np.fft.fftfreq(N, d=1/fs)
    X = np.abs(np.fft.fft(x))
    X = np.clip(X, 0, 50)  # Limitar magnitud a 50 para mejor visualización
    plt.plot(f[:N//2], X[:N//2])
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.show(block=False)

def graficar_tiempo(x, fs, titulo):
    t = np.arange(len(x)) / fs
    plt.plot(t[:int(fs*0.005)], x[:int(fs*0.005)])
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show(block=False)

def graficar_espectro_isb(lsb, usb, fs):
    N = len(lsb)
    f = np.fft.fftfreq(N, d=1/fs)
    S_lsb = np.abs(np.fft.fft(lsb))
    S_usb = np.abs(np.fft.fft(usb))
    S_lsb = np.clip(S_lsb, 0, 50)
    S_usb = np.clip(S_usb, 0, 50)
    plt.plot(f[:N//2], S_lsb[:N//2], label="LSB (audio_L_ISB)", linestyle='--')
    plt.plot(f[:N//2], S_usb[:N//2], label="USB (audio_R_ISB)", linestyle='-')
    plt.title("Espectro de ISB (LSB y USB)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def modulacion_ssb(fs, banda='USB'):
    fs, audio = wavfile.read("audio_mono.wav")
    audio = audio.astype(np.float32)
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    audio -= np.mean(audio)  # Elimina offset DC
    audio /= np.max(np.abs(audio))
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    return ssb, t, fs, audio

def modulacion_ssb_fc(audio, fs, banda='USB'):
    t = np.arange(len(audio)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    return carrier + ssb

def modulacion_isb(_, fc):
    fs_L, audio_L = wavfile.read("audio_L_ISB.wav")
    fs_R, audio_R = wavfile.read("audio_R_ISB.wav")
    assert fs_L == fs_R, "Ambos archivos deben tener la misma frecuencia de muestreo"
    fs = fs_L
    audio_L = audio_L.astype(np.float32)
    audio_R = audio_R.astype(np.float32)
    audio_L = audio_L if audio_L.ndim == 1 else audio_L.mean(axis=1)
    audio_R = audio_R if audio_R.ndim == 1 else audio_R.mean(axis=1)
    audio_L -= np.mean(audio_L)
    audio_R -= np.mean(audio_R)
    audio_L /= np.max(np.abs(audio_L))
    audio_R /= np.max(np.abs(audio_R))
    N = min(len(audio_L), len(audio_R))
    audio_L = audio_L[:N]
    audio_R = audio_R[:N]
    t = np.arange(N) / fs
    analytic_L = hilbert(audio_L)
    analytic_R = hilbert(audio_R)
    lsb = np.real(np.conj(analytic_L) * np.exp(1j * 2 * np.pi * fc * t))
    usb = np.real(analytic_R * np.exp(1j * 2 * np.pi * fc * t))
    isb = lsb + usb
    return isb, t, fs, audio_L, audio_R, lsb, usb

def ejecutar_modulacion():
    while True:
        print("\nSeleccione el tipo de modulación:")
        print("1. Modulación SSB-SC (USB)")
        print("2. Modulación SSB-SC (LSB)")
        print("3. Modulación SSB-FC (USB)")
        print("4. Modulación SSB-FC (LSB)")
        print("5. Modulación ISB")
        print("6. Salir")
        opcion = input("Ingrese el número de su selección: ")

        if opcion == '1':
            ssb, t, fs, audio = modulacion_ssb(fc, 'USB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(audio, fs, "Audio original (tiempo, 5 ms)")
            graficar_espectro(audio, fs, "Espectro del audio original (banda base)")
            graficar_tiempo(ssb, fs, "Señal AM-SSB USB (tiempo, 5 ms)")
            graficar_espectro(ssb, fs, "Espectro de la señal AM-SSB USB")

        elif opcion == '2':
            ssb, t, fs, audio = modulacion_ssb(fc, 'LSB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(audio, fs, "Audio original (tiempo, 5 ms)")
            graficar_espectro(audio, fs, "Espectro del audio original (banda base)")
            graficar_tiempo(ssb, fs, "Señal AM-SSB LSB (tiempo, 5 ms)")
            graficar_espectro(ssb, fs, "Espectro de la señal AM-SSB LSB")

        elif opcion == '3':
            fs, audio = wavfile.read("audio_mono.wav")
            audio = audio.astype(np.float32)
            audio = audio if audio.ndim == 1 else audio.mean(axis=1)
            audio -= np.mean(audio)
            audio /= np.max(np.abs(audio))
            ssb_fc = modulacion_ssb_fc(audio, fs, 'USB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb_fc, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(audio, fs, "Audio original (tiempo, 5 ms)")
            graficar_espectro(audio, fs, "Espectro del audio original (banda base)")
            graficar_tiempo(ssb_fc, fs, "Señal AM-SSB-FC USB (tiempo, 5 ms)")
            graficar_espectro(ssb_fc, fs, "Espectro de la señal AM-SSB-FC USB")

        elif opcion == '4':
            fs, audio = wavfile.read("audio_mono.wav")
            audio = audio.astype(np.float32)
            audio = audio if audio.ndim == 1 else audio.mean(axis=1)
            audio -= np.mean(audio)
            audio /= np.max(np.abs(audio))
            ssb_fc = modulacion_ssb_fc(audio, fs, 'LSB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb_fc, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(audio, fs, "Audio original (tiempo, 5 ms)")
            graficar_espectro(audio, fs, "Espectro del audio original (banda base)")
            graficar_tiempo(ssb_fc, fs, "Señal AM-SSB-FC LSB (tiempo, 5 ms)")
            graficar_espectro(ssb_fc, fs, "Espectro de la señal AM-SSB-FC LSB")

        elif opcion == '5':
            isb, t, fs, audio_L, audio_R, lsb, usb = modulacion_isb(None, fc)
            señal_total = np.concatenate((tono(7000, 0.2, fs), isb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(isb, fs, "Señal ISB (tiempo, 5 ms)")
            graficar_espectro(isb, fs, "Espectro de la señal ISB")
            graficar_espectro_isb(lsb, usb, fs)

        elif opcion == '6':
            print("Saliendo...")
            break

        else:
            print("Opción inválida. Intente de nuevo.")

        input("✅ Presione Enter para volver al menú principal...")

if __name__ == '__main__':
    ejecutar_modulacion()

