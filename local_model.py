import requests
import json
import time
import numpy as np
import pyaudio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openwakeword.model import Model as WakeWordModel
from vosk import Model, KaldiRecognizer
from piper import PiperVoice
import wave
import winsound

model = Model("vosk-model-small-en-us-0.15")
wake_model = WakeWordModel()
analyzer = SentimentIntensityAnalyzer()

print("\nLoading Piper voice...")

voice = PiperVoice.load(
    "voices/en_US-amy-medium.onnx"
)

print("Piper loaded successfully.")

RATE = 16000

WAKE_CHUNK = 1280
VOSK_CHUNK = 4096

WAKE_THRESHOLD = 0.92

COOLDOWN_SECONDS = 5

SILENCE_TIMEOUT = 3

last_activation = 0

MAX_HISTORY = 20

chat_history = []

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


def speak(text):

    try:

        with wave.open(
            "reply.wav",
            "wb"
        ) as wav_file:

            voice.synthesize_wav(
                text,
                wav_file
            )

        winsound.PlaySound(
            "reply.wav",
            winsound.SND_FILENAME
        )

    except Exception as e:

        print(
            f"\nTTS Error: {e}"
        )


def detect_emotion(text):

    score = analyzer.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.5:
        return "happy"

    elif compound <= -0.5:
        return "sad"

    else:
        return "neutral"


def ask_qwen(prompt):

    global chat_history

    emotion = detect_emotion(prompt)

    print(
        f"\nDetected Emotion: {emotion}"
    )

    chat_history.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    system_prompt = {
        "role": "system",
        "content": f"""
You are Qwen, a voice assistant.

Current user emotion: {emotion}

Rules:

- Keep replies short: 2 to 4 sentences. They are read aloud.
- If emotion is happy, respond warmly and enthusiastically.
- If emotion is sad, respond empathetically and supportively.
- If emotion is neutral, respond normally.

Keep responses natural and conversational.
"""
    }

    messages = [system_prompt]
    messages.extend(chat_history)

    try:

        response = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7
            },
            timeout=120
        )

        answer = response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:

        print("\nCannot reach llama-server. Is it running on port 8080?")
        return ""

    except requests.exceptions.Timeout:

        print("\nRequest timed out.")
        return ""

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    return answer


def wait_for_wake_word():

    global last_activation

    wake_model.reset()

    warmup_chunks = int(RATE / WAKE_CHUNK * 1.5)

    for _ in range(warmup_chunks):

        audio = stream.read(
            WAKE_CHUNK,
            exception_on_overflow=False
        )

        audio_np = np.frombuffer(
            audio,
            dtype=np.int16
        )

        wake_model.predict(
            audio_np
        )

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
        model,
        RATE
    )

    transcript = []

    session_start = time.time()
    last_speech_time = time.time()

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

            last_speech_time = time.time()

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

                last_speech_time = time.time()

                print(
                    f"\rListening: {current_partial}",
                    end=""
                )

                last_partial = current_partial

        if (
            time.time() - last_speech_time
            > SILENCE_TIMEOUT
        ):

            print(
                f"\n\n{SILENCE_TIMEOUT} seconds of silence detected."
            )

            break

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


while True:

    wait_for_wake_word()

    transcript = voice_session()

    prompt = " ".join(
        entry["text"]
        for entry in transcript
    )

    prompt = prompt.replace(
        "stop listening",
        ""
    ).strip()

    if not prompt:

        print(
            "\nNo speech captured."
        )

        continue

    print("\nCombined Prompt:")
    print(prompt)

    answer = ask_qwen(prompt)

    if answer:
        print("\nJarvis:")
        print(answer)

        try:
            speak(answer)
        except Exception:
            pass

    print(
        "\nReturning to wake mode..."
    )

    time.sleep(
        COOLDOWN_SECONDS
    )