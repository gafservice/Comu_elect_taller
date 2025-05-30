import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read

# === Parámetros ===
fs = 48000
duracion = 5
archivo_estereo = 'audio_estereo.wav'
archivo_mono = 'audio_mono.wav'

# === Función para suavizar el audio monofónico ===
def suavizar(audio, N=5):
    return np.convolve(audio, np.ones(N)/N, mode='same').astype(np.int16)

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

# === Reproducir estéreo ===
print("🔊 Reproduciendo audio estéreo...")
sd.play(audio_estereo, fs)
sd.wait()

# === Reproducir mono suavizado ===
print("🔊 Reproduciendo audio monofónico suavizado...")
sd.play(audio_mono_suave, fs)
sd.wait()

print("✅ Reproducción finalizada.")

