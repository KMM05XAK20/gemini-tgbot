import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    # печатаем только те, что поддерживают генерацию контента
    if "generateContent" in (m.supported_actions or []):
        print(m.name)
