import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3-flash-preview"  # если другой — возьми из list_models.py

def get_text(resp) -> str:
    # 1) самый удобный путь (как в доках)
    if getattr(resp, "text", None):
        return resp.text

    # 2) fallback: candidates -> content -> parts -> text
    cands = getattr(resp, "candidates", None) or []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                return t

    # 3) если реально нет текста — покажем, что пришло
    return f"[no text returned] raw={resp}"

print("Gemini CLI. Ctrl+C — выход.\n")

while True:
    try:
        q = input("you> ").strip()
        if not q:
            continue
        resp = client.models.generate_content(model=MODEL, contents=q)
        print("gemini>", get_text(resp), "\n")
    except KeyboardInterrupt:
        print("\nВыход.")
        break
