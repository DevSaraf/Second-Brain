from piper import PiperVoice
import wave
import winsound

print("Loading voice...")

voice = PiperVoice.load(
    "voices/en_US-amy-medium.onnx"
)

print("Generating speech...")

with wave.open("reply.wav", "wb") as wav_file:

    voice.synthesize_wav(
        "Hello. I am Jarvis. Nice to meet you.",
        wav_file
    )

print("Playing audio...")

winsound.PlaySound(
    "reply.wav",
    winsound.SND_FILENAME
)

print("Done.")