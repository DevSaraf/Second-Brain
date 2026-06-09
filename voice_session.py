import json
import time
import numpy as np
import pyaudio

from openwakeword.model import Model as WakeWordModel
from vosk import Model as VoskModel
from vosk import KaldiRecognizer


# --------------------------
# CONFIG
# --------------------------

RATE = 16000
WAKE_CHUNK = 1280
VOSK_CHUNK = 4096

WAKE_THRESHOLD = 0.92
COOLDOWN_SECONDS = 5

last_activation = 0


# --------------------------
# LOAD MODELS
# --------------------------

print("Loading models...")

wake_model = WakeWordModel()

print("\nLoaded Wake Word Models:")

for model_name in wake_model.models.keys():
    print("-", model_name)

vosk_model = VoskModel(
    "vosk-model-small-en-us-0.15"
)

print("\nModels loaded.\n")


# --------------------------
# AUDIO CONFIG
# --------------------------

mic = pyaudio.PyAudio()

info = mic.get_default_input_device_info()

print("\nUsing microphone:")
print(info["name"])

print("\nDefault Sample Rate:")
print(info["defaultSampleRate"])

stream = mic.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=WAKE_CHUNK
)

stream.start_stream()


# --------------------------
# WAIT FOR WAKE WORD
# --------------------------

def wait_for_wake_word():

    global last_activation

    print("\nWaiting for wake word...")
    print("Say: Hey Jarvis\n")

    while True:

        current_time = time.time()

        audio = stream.read(
            WAKE_CHUNK,
            exception_on_overflow=False
        )

        audio_np = np.frombuffer(
            audio,
            dtype=np.int16
        )

        prediction = wake_model.predict(
            audio_np
        )

        for wakeword, score in prediction.items():

            # Debug score display
            if score > 0.20:

                print(
                    f"\r{wakeword}: {score:.2f}",
                    end=""
                )

            if score > WAKE_THRESHOLD:

                # Cooldown protection
                if (
                    current_time
                    - last_activation
                    < COOLDOWN_SECONDS
                ):
                    continue

                last_activation = current_time

                print(
                    f"\n\nWake Word Detected: {wakeword}"
                )

                return


# --------------------------
# VOICE SESSION
# --------------------------

def voice_session():

    global stream

    stream.stop_stream()
    stream.close()

    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=VOSK_CHUNK
    )

    stream.start_stream()

    recognizer = KaldiRecognizer(
        vosk_model,
        RATE
    )

    transcript = []

    session_start = time.time()

    print("\nSession Started")
    print("Speak normally.")
    print("Say 'stop listening' to end.\n")

    last_partial = ""

    while True:

        data = stream.read(
            VOSK_CHUNK,
            exception_on_overflow=False
        )

        if recognizer.AcceptWaveform(data):

            result = json.loads(
                recognizer.Result()
            )

            text = result.get(
                "text",
                ""
            )

            if not text:
                continue

            elapsed = (
                time.time()
                - session_start
            )

            transcript.append(
                {
                    "timestamp": elapsed,
                    "text": text
                }
            )

            print(
                f"\n[{elapsed:.1f}s] {text}"
            )

            # ----------------
            # STOP COMMAND
            # ----------------

            if (
                "stop listening"
                in text.lower()
            ):

                print(
                    "\nStop command detected."
                )

                # Flush microphone buffer
                for _ in range(20):

                    stream.read(
                        VOSK_CHUNK,
                        exception_on_overflow=False
                    )

                break

        else:

            partial = json.loads(
                recognizer.PartialResult()
            )

            current_partial = partial.get(
                "partial",
                ""
            )

            if (
                current_partial
                and current_partial != last_partial
            ):

                print(
                    f"\rListening: {current_partial}",
                    end=""
                )

                last_partial = current_partial

    stream.stop_stream()
    stream.close()

    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=WAKE_CHUNK
    )

    stream.start_stream()

    return transcript


# --------------------------
# MAIN LOOP
# --------------------------

while True:

    wait_for_wake_word()

    transcript = voice_session()

    print("\n\nSESSION TRANSCRIPT")
    print("------------------------")

    for entry in transcript:

        print(
            f"[{entry['timestamp']:.1f}s]"
        )

        print(
            entry["text"]
        )

        print()

    print("------------------------")

    print(
        f"\nReturning to wake word mode..."
    )

    print(
        f"Cooldown: {COOLDOWN_SECONDS} seconds"
    )

    time.sleep(
        COOLDOWN_SECONDS
    )

    print()