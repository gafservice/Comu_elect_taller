import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read

# === Parámetros ===
fs = 48000
duracion = 5
archivo_estereo = 'audio_estereo.wav'
archivo_mono = 'audio_mono.wav'
archivo_L_ISB = 'audio_L_ISB.wav'  # Canal izquierdo para ISB
archivo_R_ISB = 'audio_R_ISB.wav'  # Canal derecho para ISB
frecuencia_tono_L = 6000  # Tono de 6000 Hz para canal L
frecuencia_tono_R = 8000  # Tono de 8000 Hz para canal R
amplitud_tono = 512 # Mitad de la amplitud de 8191 (8191 / 2)

# === Función para suavizar el audio monofónico ===
def suavizar(audio, N=5):
    return np.convolve(audio, np.ones(N)/N, mode='same').astype(np.int16)

# === Generar tono a una frecuencia dada ===
def generar_tono(frecuencia, duracion, fs):
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)  # Tiempo de la señal
    tono = amplitud_tono * np.sin(2 * np.pi * frecuencia * t)  # Tono con amplitud reducida
    return tono.astype(np.int16)

# === Intentar usar PulseAudio ===
dispositivo = 'pulse'
try:
    sd.check_input_settings(device=dispositivo, channels=2, samplerate=fs)
    print(f"🎯 Usando dispositivo PulseAudio ('{dispositivo}') con 2 canales.")
except Exception as e:
    print("⚠️ PulseAudio no disponible o no soporta estéreo. Buscando dispositivo manualmente...")
    dispositivo = None
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] >= 2:
            dispositivo = i
            print(f"✅ Usando dispositivo alternativo: {d['name']} (índice {i})")
            break
    if dispositivo is None:
        raise RuntimeError("❌ No se encontró un dispositivo estéreo válido.")

# === Grabar en estéreo ===
print(f"🎙️ Grabando {duracion} segundos en estéreo a {fs} Hz...")
audio_estereo = sd.rec(int(duracion * fs), samplerate=fs, channels=2, dtype='int16', device=dispositivo)
sd.wait()
print("✅ Grabación completada.")
print("🔍 Primeras muestras (L y R):\n", audio_estereo[:5])

# === Guardar archivo estéreo ===
write(archivo_estereo, fs, audio_estereo)
print(f"💾 Audio estéreo guardado como: {archivo_estereo}")

# === Convertir a mono y suavizar ===
audio_mono = np.mean(audio_estereo, axis=1).astype(np.int16)
audio_mono_suave = suavizar(audio_mono)

# === Guardar archivo monofónico ===
write(archivo_mono, fs, audio_mono_suave)
print(f"💾 Audio monofónico suavizado guardado como: {archivo_mono}")

# === Aplicar efectos para ISB ===
# Canal izquierdo: Generar tono de 2500 Hz y agregarlo al canal izquierdo (L)
audio_L_ISB = audio_estereo[:, 0]  # Canal izquierdo sin efectos adicionales
tono_L = generar_tono(frecuencia_tono_L, duracion, fs)
audio_L_ISB_con_tono = np.clip(audio_L_ISB + tono_L[:len(audio_L_ISB)], -32768, 32767)  # Agregar tono

# Guardar el archivo con tono
write(archivo_L_ISB, fs, audio_L_ISB_con_tono)
print(f"💾 Audio L (ISB) guardado con tono de 2500 Hz como: {archivo_L_ISB}")

# Canal derecho: Generar tono de 5000 Hz y agregarlo al canal derecho (R)
audio_R_ISB = audio_estereo[:, 1]  # Canal derecho sin efectos adicionales
tono_R = generar_tono(frecuencia_tono_R, duracion, fs)
audio_R_ISB_con_tono = np.clip(audio_R_ISB + tono_R[:len(audio_R_ISB)], -32768, 32767)  # Agregar tono

# Guardar el archivo con tono
write(archivo_R_ISB, fs, audio_R_ISB_con_tono)
print(f"💾 Audio R (ISB) guardado con tono de 5000 Hz como: {archivo_R_ISB}")

# === Reproducir estéreo ===
print("🔊 Reproduciendo audio estéreo...")
sd.play(audio_estereo, fs)
sd.wait()

# === Reproducir mono suavizado ===
print("🔊 Reproduciendo audio monofónico suavizado...")
sd.play(audio_mono_suave, fs)
sd.wait()

# === Reproducir audio L (ISB) con tono de 6000 Hz ===
print("🔊 Reproduciendo audio L (ISB) con tono de 6000 Hz...")
sd.play(audio_L_ISB_con_tono, fs)
sd.wait()

# === Reproducir audio R (ISB) con tono de 8000 Hz ===
print("🔊 Reproduciendo audio R (ISB) con tono de 8000 Hz...")
sd.play(audio_R_ISB_con_tono, fs)
sd.wait()

print("✅ Reproducción finalizada.")

