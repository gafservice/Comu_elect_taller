import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar errores en hilos
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, lfilter, resample
from scipy.io.wavfile import write
import time
import os

# Filtro pasa bajas Butterworth.
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    y = lfilter(b, a, data)
    return y

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
    blocksize = int(0.1 * fs)
    dur_max_mensaje = 10
    umbral_inicio = 10
    umbral_fin = 5

    # Error de fase.
    phi = np.pi
    # Error de frecuencia.
    deltaf = 0.01 * fc

    sd.default.samplerate = fs
    sd.default.channels = 1

    print("🔁 Sistema activo. Esperando tono de 4000 Hz...")

    while True:
        print("🕑 Esperando 0.5 segundos antes de iniciar...")
        time.sleep(0.5)

        mensaje = []
        inicio_detectado = False
        espectro_guardado = False
        print("🎧 Escuchando en tiempo real...")

        def callback_inicial(indata, frames, time_info, status):
            nonlocal inicio_detectado, espectro_guardado, mensaje
            bloque = indata[:, 0]
            detectado, f, S = detectar_tono(bloque, 4000, fs, margen=30, umbral=umbral_inicio)
            if detectado and not inicio_detectado:
                inicio_detectado = True
                mensaje.append(bloque.copy())
                print("✅ Tono de inicio detectado.")
                if not espectro_guardado:
                    nombre_png = siguiente_nombre().replace('.wav', '_espectro.png')
                    plt.figure()
                    plt.plot(f, S)
                    plt.title("Espectro al detectar el tono de inicio (2000 Hz)")
                    plt.xlabel("Frecuencia [Hz]")
                    plt.ylabel("Magnitud")
                    plt.grid()
                    plt.savefig(nombre_png)
                    plt.close()
                    espectro_guardado = True
                    print(f"🖼️ Espectro guardado como '{nombre_png}'")
            elif inicio_detectado:
                mensaje.append(bloque.copy())

        with sd.InputStream(callback=callback_inicial, blocksize=blocksize):
            while not inicio_detectado:
                time.sleep(0.05)

            if not inicio_detectado:
                print("⌛ No se detectó tono de inicio. Reiniciando...\n")
                continue

        print("⏺️ Grabando mensaje hasta detectar tono de fin (5000 Hz)...")

        def callback_mensaje(indata, frames, time_info, status):
            nonlocal mensaje
            bloque = indata[:, 0]
            detectado_fin, _, _ = detectar_tono(bloque, 5000, fs, margen=30, umbral=umbral_fin)
            if not detectado_fin:
                mensaje.append(bloque.copy())
            else:
                print("✅ Tono de fin detectado.")
                raise sd.CallbackStop()

        with sd.InputStream(callback=callback_mensaje, blocksize=blocksize):
            try:
                sd.sleep(int(dur_max_mensaje * 1000))
            except sd.CallbackStop:
                pass

        mensaje = np.concatenate(mensaje)
        t = np.arange(len(mensaje)) / fs

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
        print(f"🖼️ Señal en el tiempo guardada como '{nombre_senal}'")

        output_file = nombre_senal.replace('_tiempo.png', '.wav')
        print(f"🔊 Reproduciendo mensaje demodulado...")
        sd.play(audio, fs)
        sd.wait()
        write(output_file, fs, (audio * 32767).astype(np.int16))
        print(f"💾 Audio guardado como '{output_file}'\n")

        print("🔁 Reiniciando escucha...\n")

if __name__ == '__main__':
    main()
