import json
import time
import numpy as np
import pyaudio

from openwakeword.model import Model as WakeWordModel
from vosk import Model as VoskModel
from vosk import KaldiRecognizer


# --------------------------
# LOAD MODELS
# --------------------------

print("Loading models...")

wake_model = WakeWordModel()

vosk_model = VoskModel(
    "vosk-model-small-en-us-0.15"
)

print("Models loaded.\n")


# --------------------------
# AUDIO CONFIG
# --------------------------

RATE = 16000
CHUNK = 1280

mic = pyaudio.PyAudio()

stream = mic.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

stream.start_stream()


# --------------------------
# WAIT FOR WAKE WORD
# --------------------------

def wait_for_wake_word():

    print("Waiting for wake word...")
    print("Say: Hey Jarvis\n")

    while True:

        audio = stream.read(
            CHUNK,
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

            if score > 0.5:

                print(
                    f"\nWake Word Detected: {wakeword}"
                )

                return


# --------------------------
# VOICE SESSION
# --------------------------

def voice_session():

    recognizer = KaldiRecognizer(
        vosk_model,
        RATE
    )

    transcript = []

    session_start = time.time()

    print("\nSession Started")
    print(
        "Speak normally."
    )
    print(
        "Say 'stop listening' to end.\n"
    )

    last_partial = ""

    while True:

        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        # --------------------
        # LIVE TRANSCRIPT
        # --------------------

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

        # --------------------
        # FINAL TRANSCRIPT
        # --------------------

        if recognizer.AcceptWaveform(
            data
        ):

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

            if (
                "stop listening"
                in text.lower()
            ):

                print(
                    "\nStop command detected."
                )

                break

    return transcript


# --------------------------
# MAIN LOOP
# --------------------------

while True:

    wait_for_wake_word()

    transcript = voice_session()

    print(
        "\n\nSESSION TRANSCRIPT"
    )

    print(
        "------------------------"
    )

    for entry in transcript:

        print(
            f"[{entry['timestamp']:.1f}s]"
        )

        print(
            entry["text"]
        )

        print()

    print(
        "------------------------"
    )

    print(
        "\nReturning to wake word mode...\n"
    )