import requests

while True:

    prompt = input("User: ")

    if prompt.lower() == "exit":
        break

    response = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "system",
                    "content": """
You are Qwen.

Rules:

1. Answer only mathematics, coding, DSA and computer science questions.

2. For mathematics questions:
   First line: answer only.
   Second line: short explanation.

3. For coding questions:
   Give detailed explanations.

4. If the question is unrelated, reply exactly:
   "I can only answer mathematics and programming related questions."

5. Respond only in English.
"""
                },
                {
                    
                    "role": "user",
                    "content": prompt
                }
            ],
            
        }
    )

    answer = response.json()["choices"][0]["message"]["content"]

    print("Chatbot:", answer)