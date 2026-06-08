import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def detect_emotion(text):

    score = analyzer.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.5:
        return "happy"

    elif compound <= -0.5:
        return "sad"

    else:
        return "neutral"

chat_history = []

MAX_HISTORY = 20


while True:

    prompt = input("\nUser: ")

    if prompt.lower() == "exit":
        break


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