import os
import asyncio
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

from google import genai

load_dotenv("/opt/gemini/.env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Gemini client
client = genai.Client(api_key=GEMINI_KEY)

def extract_text(resp) -> str:
    # быстрый путь
    if getattr(resp, "text", None):
        return resp.text

    # fallback на candidates.parts
    cands = getattr(resp, "candidates", None) or []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                return t
    return "Не смог сформировать ответ (пустой ответ от модели)."

async def ask_gemini(prompt: str) -> str:
    # ВАЖНО: generate_content синхронный — уводим в thread, чтобы не блокировать бота
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model=MODEL,
        contents=prompt
    )
    return extract_text(resp)

dp = Dispatcher()

@dp.message(F.text == "/start")
async def start(m: Message):
    await m.answer("Привет! Напиши вопрос — я спрошу Gemini и отвечу.")

@dp.message(F.text)
async def handle_text(m: Message):
    q = m.text.strip()
    if not q:
        return
    await m.chat.do("typing")
    try:
        answer = await ask_gemini(q)
        # Telegram ограничивает длину сообщения; подрежем аккуратно
        if len(answer) > 3800:
            answer = answer[:3800] + "…"
        await m.answer(answer, parse_mode="HTML")
    except Exception as e:
        await m.answer(f"Ошибка при обращении к Gemini: {type(e).__name__}: {e}")

async def main():
    bot = Bot(token=BOT_TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
