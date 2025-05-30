import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.pyplot as plt

fc = 10000  # Frecuencia fija de la portadora

def tono(frec, dur, fs):
    t = np.arange(0, int(fs * dur)) / fs
    return 0.7 * np.sin(2 * np.pi * frec * t)

def graficar_espectro(x, fs, titulo):
    N = len(x)
    f = np.fft.fftfreq(N, d=1/fs)
    X = np.abs(np.fft.fft(x))
    X = np.clip(X, 0, 50)
    plt.figure()
    plt.plot(f[:N//2], X[:N//2])
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.show(block=False)

def graficar_tiempo(x, fs, titulo):
    t = np.arange(len(x)) / fs
    plt.figure()
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
    plt.figure()
    plt.plot(f[:N//2], S_lsb[:N//2], label="LSB (audio_L_ISB)", linestyle='--')
    plt.plot(f[:N//2], S_usb[:N//2], label="USB (audio_R_ISB)", linestyle='-')
    plt.title("Espectro combinado de ISB")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def cargar_audio_mono():
    fs, audio = wavfile.read("audio_mono.wav")
    audio = audio.astype(np.float32)
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    audio -= np.mean(audio)
    audio /= np.max(np.abs(audio))
    return fs, audio

def modulacion_ssb(audio, fs, banda='USB'):
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    return ssb

def modulacion_ssb_fc(audio, fs, banda='USB'):
    t = np.arange(len(audio)) / fs
    analytic = hilbert(audio)
    if banda == 'USB':
        ssb = np.real(analytic * np.exp(1j * 2 * np.pi * fc * t))
    else:
        ssb = np.real(np.conj(analytic) * np.exp(1j * 2 * np.pi * fc * t))
    carrier = np.cos(2 * np.pi * fc * t)
    return carrier + ssb

def modulacion_isb():
    fs_L, audio_L = wavfile.read("audio_L_ISB.wav")
    fs_R, audio_R = wavfile.read("audio_R_ISB.wav")
    assert fs_L == fs_R
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
    t = np.arange(N) / fs
    analytic_L = hilbert(audio_L[:N])
    analytic_R = hilbert(audio_R[:N])
    lsb = np.real(np.conj(analytic_L) * np.exp(1j * 2 * np.pi * fc * t))
    usb = np.real(analytic_R * np.exp(1j * 2 * np.pi * fc * t))
    isb = lsb + usb
    return isb, fs, lsb, usb

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

        if opcion in ['1', '2', '3', '4']:
            fs, audio = cargar_audio_mono()

        if opcion == '1':
            ssb = modulacion_ssb(audio, fs, 'USB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(ssb, fs, "AM-SSB-SC USB (tiempo)")
            graficar_espectro(ssb, fs, "AM-SSB-SC USB (frecuencia)")

        elif opcion == '2':
            ssb = modulacion_ssb(audio, fs, 'LSB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(ssb, fs, "AM-SSB-SC LSB (tiempo)")
            graficar_espectro(ssb, fs, "AM-SSB-SC LSB (frecuencia)")

        elif opcion == '3':
            ssb_fc = modulacion_ssb_fc(audio, fs, 'USB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb_fc, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(ssb_fc, fs, "AM-SSB-FC USB (tiempo)")
            graficar_espectro(ssb_fc, fs, "AM-SSB-FC USB (frecuencia)")

        elif opcion == '4':
            ssb_fc = modulacion_ssb_fc(audio, fs, 'LSB')
            señal_total = np.concatenate((tono(7000, 0.2, fs), ssb_fc, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(ssb_fc, fs, "AM-SSB-FC LSB (tiempo)")
            graficar_espectro(ssb_fc, fs, "AM-SSB-FC LSB (frecuencia)")

        elif opcion == '5':
            isb, fs, lsb, usb = modulacion_isb()
            señal_total = np.concatenate((tono(7000, 0.2, fs), isb, tono(5000, 0.2, fs)))
            sd.play(señal_total, fs, blocking=True)
            graficar_tiempo(isb, fs, "Señal ISB (tiempo)")
            graficar_espectro(isb, fs, "Espectro total ISB")
            graficar_espectro_isb(lsb, usb, fs)

        elif opcion == '6':
            print("Saliendo...")
            break

        else:
            print("Opción inválida.")

        input("✅ Presione Enter para volver al menú principal...")

if __name__ == '__main__':
    ejecutar_modulacion()

