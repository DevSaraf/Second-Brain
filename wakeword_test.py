# wakeword_test.py

import pyaudio
import numpy as np
from openwakeword.model import Model

model = Model()

mic = pyaudio.PyAudio()

stream = mic.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("Listening for wake word...")

while True:

    audio = stream.read(
        1280,
        exception_on_overflow=False
    )

    audio = np.frombuffer(
        audio,
        dtype=np.int16
    )

    prediction = model.predict(audio)

    for wakeword, score in prediction.items():

        if score > 0.5:

            print(
                f"\nWake Word Detected: {wakeword}"
            )

            break