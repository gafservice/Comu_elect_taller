import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read

# === Parámetros ===
fs = 44100                      # Frecuencia de muestreo (Hz)
duracion = 5                    # Duración en segundos
archivo_estereo = 'audio.wav'  # Archivo original
archivo_mono = 'audio_mono.wav' # Archivo monofónico

# === Buscar primer dispositivo válido ===
dispositivo = None
canales = 1

for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] >= 1:
        dispositivo = i
        canales = d['max_input_channels']
        print(f"✅ Usando dispositivo {i}: {d['name']} con {canales} canal(es)")
        break

if dispositivo is None:
    raise RuntimeError("❌ No se encontró un dispositivo de entrada válido.")

# === Grabación ===
print(f"🎙️ Grabando {duracion} segundos desde el micrófono con {canales} canal(es)...")
audio = sd.rec(int(duracion * fs), samplerate=fs, channels=canales, dtype='int16', device=dispositivo)
sd.wait()
print("✅ Grabación terminada.")

print("🔍 Primeros valores grabados:", audio[:10].flatten())




# === Guardar archivo original ===
write(archivo_estereo, fs, audio)
print(f"💾 Audio original guardado como: {archivo_estereo}")

# === Convertir a mono ===
if canales > 1:
    print("🔄 Convirtiendo a monofónico...")
    audio_mono = np.mean(audio, axis=1).astype(np.int16)
else:
    audio_mono = audio
print(f"✅ Conversión a mono completada.")

# === Guardar archivo mono ===
write(archivo_mono, fs, audio_mono)
print(f"💾 Audio monofónico guardado como: {archivo_mono}")

# === Reproducir ambos ===
print("🔊 Reproduciendo audio original...")
sd.play(audio, fs)
sd.wait()

print("🔊 Reproduciendo audio monofónico...")
sd.play(audio_mono, fs)
sd.wait()

print("✅ Reproducción finalizada.")

