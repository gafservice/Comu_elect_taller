import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar errores en hilos
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import butter, filtfilt, hilbert, lfilter, resample
from scipy.io.wavfile import write
import time
import os

def butter_lowpass(cutoff, fs, order=6):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return b, a
# Filtro pasa bajas Butterworth.
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    y = lfilter(b, a, data)
    return y

def detectar_tono(bloque, tono, fs, margen=30, umbral=10):
    N = len(bloque)
@@ -31,6 +34,59 @@ def siguiente_nombre():
        i += 1
    return f"{base}{i:03d}.wav"

def ssb_demodulate(signal, fc, fs, sideband='usb', is_fc=False):
    """
    Demodula una señal SSB.
    Parámetros:
    - signal: señal modulada (SSB-SC o SSB-FC)
    - fc: frecuencia de la portadora
    - fs: frecuencia de muestreo
    - sideband: 'usb' o 'lsb'
    - is_fc: True si es SSB-FC, False si es SSB-SC
    Retorna:
    - Señal demodulada (audio base)
    """

    t = np.arange(len(signal)) / fs

    # Si es SSB-FC, primero eliminar la portadora (puedes usar un filtro notch o restarla si se conoce)
    if is_fc:
        # Estimar y remover la portadora (asumimos Ac=1)
        carrier = np.cos(2 * np.pi * fc * t)
        signal = signal - carrier

    # Demodulación coherente
    if sideband == 'usb':
        demodulated = signal * np.cos(2 * np.pi * (fc + deltaf) * t + phi)
    elif sideband == 'lsb':
        demodulated = signal * np.cos(2 * np.pi * (fc + deltaf) * t + phi)
    else:
        raise ValueError("sideband debe ser 'usb' o 'lsb'")

    # Filtro pasa bajos para recuperar el mensaje
    cutoff = 20000  # Frecuencia de corte para filtro pasa bajos
    recovered = butter_lowpass_filter(demodulated, cutoff=cutoff, fs=fs, order=6)

    return recovered

def envelope_demodulation(signal, fs, cutoff=20000):
    """
    Demodulación por detección de envolvente con filtrado pasa bajos.
    """
    # Obtener la envolvente con Hilbert
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Filtro pasa bajas adecuado
    envelope_filtered = butter_lowpass_filter(envelope, cutoff=cutoff, fs=fs, order=10)

    envelope_filtered = envelope_filtered - np.mean(envelope_filtered)
    envelope_filtered = envelope_filtered / np.max(np.abs(envelope_filtered))

    return envelope_filtered

def main():
    fs = 44100
    fc = 10000
    umbral_inicio = 10
    umbral_fin = 5

    # Error de fase.
    phi = np.pi
    # Error de frecuencia.
    deltaf = 0.01 * fc

    sd.default.samplerate = fs
    sd.default.channels = 1

@@ -104,18 +165,60 @@ def callback_mensaje(indata, frames, time_info, status):

        mensaje = np.concatenate(mensaje)
        t = np.arange(len(mensaje)) / fs
        portadora = np.cos(2 * np.pi * fc * t)
        baseband = mensaje * portadora
        b, a = butter_lowpass(4000, fs)
        audio = filtfilt(b, a, baseband)
        audio /= np.max(np.abs(audio))

        plt.figure()
        plt.plot(t, audio)
        plt.title("Mensaje demodulado (dominio del tiempo)")
        plt.xlabel("Tiempo [s]")

        #portadora = np.cos(2 * np.pi * fc * t)
        #baseband = mensaje * portadora
        #b, a = butter_lowpass(4000, fs)
        #audio = filtfilt(b, a, baseband)
        #audio /= np.max(np.abs(audio))

        # Demodulación coherente.
        #Caso SSB-SC.
        audio_rec_sc_usb = ssb_demodulate(mensaje, fc, fs, sideband='usb', is_fc=False)
        audio_rec_sc_lsb = ssb_demodulate(mensaje, fc, fs, sideband='lsb', is_fc=False)
        #Caso ISB.
        audio_rec_isb_usb = ssb_demodulate(mensaje, fc, fs, sideband='usb', is_fc=False)
        audio_rec_isb_lsb = ssb_demodulate(mensaje, fc, fs, sideband='lsb', is_fc=False)

        # Demodulacion por deteccion de envolvente.
        #Caso SSB-FC.
        audio_rec_fc_usb = envelope_demodulation(mensaje, fs)
        audio_rec_fc_lsb = envelope_demodulation(mensaje, fs)

        # Eleccion del tipo de demodulacion.
        demodulada = audio_rec_isb_usb
        demodulada_fft = np.abs(np.fft.fft(demodulada))
        demodulada_fft_db = 20 * np.log10(demodulada_fft)

        #plt.figure()
        #plt.plot(t, demodulada)
        #plt.title("Mensaje demodulado (dominio del tiempo)")
        #plt.xlabel("Tiempo [s]")
        #plt.ylabel("Amplitud")
        #plt.grid()

        # Graficar.
        plt.figure(figsize=(20, 16))

        # Audio de entrada en el dominio del tiempo.
        plt.subplot(2, 1, 1)
        plt.plot(t, demodulada)
        plt.title("Gráfica del audio demodulado en el tiempo")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.grid()

        # Espectro del audio de entrada.
        plt.subplot(2, 1, 2)
        plt.plot(f[:N//2], demodulada_fft_db[:N//2])
        plt.title("Espectro del audio demodulado")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud (dB)")
        plt.grid()

        plt.tight_layout()
        plt.show()

        nombre_senal = siguiente_nombre().replace('.wav', '_tiempo.png')
        plt.savefig(nombre_senal)
        plt.close()
