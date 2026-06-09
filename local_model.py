import requests
import json
import pyaudio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vosk import Model, KaldiRecognizer

model = Model("vosk-model-small-en-us-0.15")
analyzer = SentimentIntensityAnalyzer()

MAX_HISTORY = 20


def detect_emotion(text):

    score = analyzer.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.5:
        return "happy"

    elif compound <= -0.5:
        return "sad"

    else:
        return "neutral"


def listen(seconds=7):

    recognizer = KaldiRecognizer(model, 16000)

    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8192
    )
    stream.start_stream()
    print("Listening... (speak now)")

    text = ""
    for _ in range(int(16000 / 8192 * seconds)):
        data = stream.read(8192, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            text += json.loads(recognizer.Result()).get("text", "") + " "

    text += json.loads(recognizer.FinalResult()).get("text", "")

    stream.stop_stream()
    stream.close()
    mic.terminate()

    return text.strip()


chat_history = []

while True:

    print("\nChoose Input Method")
    print("1. Text")
    print("2. Voice")
    print("Type 'exit' to quit")

    choice = input("\nEnter choice: ").strip().lower()

    if choice == "exit":
        break

    elif choice == "1":

        prompt = input("\nUser: ")

        if prompt.lower() == "exit":
            break

    elif choice == "2":

        prompt = listen()

        print("You said:", prompt)

        if not prompt:
            print("(heard nothing, try again)")
            continue

    else:

        print("Invalid choice.")
        continue


    emotion = detect_emotion(prompt)

    print("Detected Emotion:", emotion)

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
You are Qwen.

Current user emotion: {emotion}

Rules:

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
                "messages": messages
            },
            timeout=60
        )

        answer = response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:

        print("\nRequest timed out.")
        continue

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print("\nChatbot:", answer)