import os
import re
import time
import html
import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

from google import genai
from google.genai import types

from storage import Storage

# -------------------- CONFIG --------------------

storage = Storage()

load_dotenv("/opt/gemini/.env")  # подстрой путь если нужно

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_IDS = os.getenv("1190756443")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

# Сколько сообщений хранить в контексте на пользователя (в каждую сторону)
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))  # 10 пар user+bot ~= 20 сообщений

# Rate limit: N запросов за WINDOW секунд
RATE_N = int(os.getenv("RATE_N", "8"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))

# Глобальный лимит параллельных запросов к Gemini
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))

# Таймаут на один запрос к Gemini
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "45"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — полезный ассистент в Telegram. "
    "Не утверждай про сервер/ОС/окружение, если тебе это явно не передали. "
    "Не проси и не раскрывай секреты (ключи, токены, .env). "
    "Пиши кратко и по делу."
)

SUMMARY_EVERY = int(os.getenv("SUMMARY_EVERY", "20"))   # раз в 20 user-сообщений
SUMMARY_LAST_N = int(os.getenv("SUMMARY_LAST_N", "60")) # сколько последних сообщений учитывать

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("tg-gemini-bot")

# -------------------- GEMINI CLIENT --------------------
client = genai.Client(api_key=GEMINI_KEY)
sem = asyncio.Semaphore(MAX_CONCURRENCY)

# -------------------- STATE (in-memory) --------------------
@dataclass
class UserState:
    model: str
    history: Deque[Tuple[str, str]]  # ("user"|"model", text)
    rate: Deque[float]              # timestamps

users: Dict[int, UserState] = {}

def get_user(uid: int) -> UserState:
    st = users.get(uid)
    if not st:
        st = UserState(model=DEFAULT_MODEL, history=deque(maxlen=MAX_TURNS * 2), rate=deque())
        users[uid] = st
    return st

# -------------------- HELPERS --------------------
def rate_limit_ok(st: UserState) -> bool:
    now = time.time()
    # удаляем старые
    while st.rate and now - st.rate[0] > RATE_WINDOW:
        st.rate.popleft()
    if len(st.rate) >= RATE_N:
        return False
    st.rate.append(now)
    return True

def md_bold_to_html(text: str) -> str:
    """
    Безопасный рендер:
    - сначала экранируем HTML
    - затем конвертим **bold** -> <b>bold</b>
    """
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe

def tg_split(text: str, limit: int = 3800) -> List[str]:
    # Telegram 4096, но оставим запас
    if len(text) <= limit:
        return [text]
    parts = []
    cur = 0
    while cur < len(text):
        parts.append(text[cur:cur + limit])
        cur += limit
    return parts

def build_contents(history: list[tuple[str, str]], user_text: str):
    """
    history: список кортежей (role, text), где role in {"user","model"}.
    user_text: текущий текст пользователя
    """
    contents: list[types.Content] = []

    for role, text in history:
        role_norm = "model" if role in ("model", "assistant") else "user"
        contents.append(
            types.Content(
                role=role_norm,
                parts=[types.Part.from_text(text=text)],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)],
        )
    )

    return contents

# def build_contents(st: UserState, user_text: str):
#     """
#     Собираем контекст в формат, понятный Gemini:
#     contents = [system + история + текущий запрос]
#     """
#     contents = []
#     contents.append(SYSTEM_PROMPT)

#     for role, txt in st.history:
#         # txt уже простой текст
#         contents.append(f"{role}: {txt}")

#     contents.append(f"user: {user_text}")
#     return contents

async def gemini_generate(model: str, contents, system_instruction: str) -> str:
    """
    Вызов Gemini в отдельном thread, с таймаутом, семафором и ретраями.
    """
    async with sem:
        for attempt in range(1, 4):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )

                resp = await asyncio.to_thread(
                    client.model.generate_content,
                    contents=contents,
                    config=config,
                )
                # resp.text иногда None -> достаём руками
                text = getattr(resp, "text", None)
                if not text:
                    return text

                fc = getattr(resp, "function_calls", None)
                if fc:
                    return "Модель попыталась вызвать инструмент (function calling), но в боте это отключено/не поддерживается."

                cands = getattr(resp, "candidates", None) or []
                for c in cands:
                    content = getattr(c, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", None) or []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            return t
                try:
                    dump = resp.model_dump()
                except Exception:
                    dump = str(resp)

                log.warning("Gemini returned empty response: %s", dump)
                return "Пустой ответ от модели."
            except asyncio.TimeoutError:
                log.warning("Gemini timeout (attempt %s)", attempt)
                if attempt == 3:
                    return "Таймаут ответа от Gemini. Попробуй ещё раз."
            except Exception as e:
                log.exception("Gemini error (attempt %s): %s", attempt, e)
                if attempt == 3:
                    return f"Ошибка Gemini: {type(e).__name__}"
                await asyncio.sleep(0.8 * attempt)



async def use_ai(uid: int) -> bool:
    
    return True

# -------------------- AIROGRAM --------------------
dp = Dispatcher()

@dp.message(F.text == "/start")
async def start(m: Message):
    st = get_user(m.from_user.id)
    await storage.ensure_user(st, m.from_user.username, m.from_user.first_name, DEFAULT_MODEL)
    await storage.log_event(st, "start")
    await storage.touch_user(st)

    await m.answer(
        "Привет! Пиши вопрос — отвечу через Gemini.\n"
        "Команды:\n"
        "/reset — сбросить контекст\n"
        "/model — показать текущую модель\n"
        "/model <name> — установить модель\n"
        "/help — помощь"
    )

@dp.message(F.text == "/help")
async def help_(m: Message):
    await m.answer(
        "Я отвечаю через Gemini.\n"
        "Команды:\n"
        "/reset — очистить контекст\n"
        "/model — текущая модель\n"
        "/model <name> — сменить модель\n\n"
        f"Лимит: {RATE_N} запросов за {RATE_WINDOW} сек."
    )

@dp.message(F.text == "/reset")
async def reset(m: Message):
    await storage.ctx_clear(m.from_user.id)
    await m.answer("Ок, контекст сброшен (быстрая память).")

@dp.message(F.text == "/forget")
async def forget(m: Message):
    uid = m.from_user.id
    await storage.ctx_clear(uid)
    await storage.clear_long_memory(uid)
    await m.answer("Ок, удалил память (summary + факты) и быстрый контекст.")


@dp.message(F.text.startswith("/remember "))
async def remember(m: Message):
    fact = m.text.split(" ", 1)[1].strip()
    if not fact:
        await m.answer("Напиши так: /remember я люблю питон")
        return
    await storage.add_fact(m.from_user.id, fact, confidence=80)
    await m.answer("Запомнил ✅")

@dp.message(F.text == "/memory")
async def memory(m: Message):
    uid = m.from_user.id
    summary = await storage.get_summary(uid) or "—"
    facts = await storage.list_facts(uid, limit=10)
    txt = "Память:\n" + summary + "\n\nФакты:\n" + ("\n".join(f"- {f}" for f in facts) if facts else "—")
    await m.answer(txt)


async def update_summary_if_needed(uid: int, model: str):
    # считаем только user-сообщения
    n = await storage.inc_user_msg_count(uid)
    if n % SUMMARY_EVERY != 0:
        return

    old = await storage.get_summary(uid) or ""
    last = await storage.last_messages(uid, limit=SUMMARY_LAST_N)

    # краткий, строгий промпт
    prompt_contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text=(
                    "Обнови краткую память диалога. "
                    "Сохраняй только устойчивые факты, цели, предпочтения пользователя, "
                    "и текущий контекст задач. "
                    "Не добавляй ничего от себя. "
                    "Пиши коротко, пунктами.\n\n"
                    f"Текущая память:\n{old}\n\n"
                    "Новые сообщения (хронологически):\n" +
                    "\n".join([f"{r}: {t}" for r, t in last])
                )
            )]
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction="Ты — модуль памяти. Твоя задача: кратко обновлять summary.",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    resp = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt_contents,
        config=config,
    )

    new_sum = getattr(resp, "text", None) or ""
    new_sum = new_sum.strip()
    if new_sum:
        await storage.set_summary(uid, new_sum)
        await storage.ctx_clear(uid)  # важное: сбрасываем Redis контекст



@dp.message(F.text.startswith("/model"))
async def model_cmd(m: Message):
    st = get_user(m.from_user.id)
    parts = m.text.split(maxsplit=1)
    if len(parts) == 1:
        await m.answer(f"Текущая модель: <code>{html.escape(st.model)}</code>", parse_mode=ParseMode.HTML)
        return
    new_model = parts[1].strip()
    st.model = new_model
    await m.answer(f"Ок, модель: <code>{html.escape(st.model)}</code>", parse_mode=ParseMode.HTML)


@dp.message(F.text == "/stats")
async def stats(m: Message):
    if m.from_user.id not in ADMIN_IDS:
        return

    async with storage.mysql_pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM users")
            users_total = (await cur.fetchone())[0]

            await cur.execute(
                "SELECT COUNT(*) FROM users WHERE last_seen > NOW() - INTERVAL 1 DAY"
            )
            dau = (await cur.fetchone())[0]

            await cur.execute(
                "SELECT COUNT(*) FROM users WHERE last_seen > NOW() - INTERVAL 7 DAY"
            )
            wau = (await cur.fetchone())[0]

            await cur.execute(
                "SELECT COUNT(*) FROM users WHERE last_seen > NOW() - INTERVAL 30 DAY"
            )
            mau = (await cur.fetchone())[0]

    await m.answer(
        f"📊 Статистика:\n"
        f"👤 Всего пользователей: {users_total}\n"
        f"🔥 DAU: {dau}\n"
        f"📈 WAU: {wau}\n"
        f"🌍 MAU: {mau}"
    )

@dp.message(F.text)
async def handle_text(m: Message):
    uid = m.from_user.id
    await storage.touch_user(uid)
    q = (m.text or "").strip()
    if not q:
        return

    # ACK чтобы бот не "молчал"
    ack = await m.answer("Думаю…")

    ok = await storage.rate_limit_ok(uid, RATE_N, RATE_WINDOW)
    if not ok:
        await ack.edit_text("Слишком часто. Подожди немного 🙂")
        return

    if not await storage.acquire_lock(uid, ttl_sec=30):
        await ack.edit_text("Подожди, я ещё отвечаю 🙂")
        return

    try:
        model = DEFAULT_MODEL
        await storage.ensure_user(uid, m.from_user.username, m.from_user.first_name, model)

        ctx = await storage.ctx_get(uid)
        contents = build_contents(ctx, q)
        summary = await storage.get_summary(uid)
        facts = await storage.list_facts(uid, limit=10)


        # ВАЖНО: обернём Gemini в таймаут
        try:
            answer = await asyncio.wait_for(
                gemini_generate(model, contents, system_plus_memory),
                timeout=GEMINI_TIMEOUT + 5
            )
        except asyncio.TimeoutError:
            answer = "Таймаут ответа от Gemini. Попробуй ещё раз."

        await storage.save_message(uid, "user", q)
        await storage.save_message(uid, "model", answer)
        await storage.ctx_append(uid, "user", q)
        await storage.ctx_append(uid, "model", answer)
        await storage.inc_messages(uid)
        await storage.log_event(uid, "message")

        await ack.edit_text(answer)

    except Exception:
        log.exception("handle_text failed")
        await ack.edit_text("Упс, ошибка на сервере. Я уже в логах 🙂")

    finally:
        await storage.release_lock(uid)
        update_summary_if_needed(uid, model)


async def main():
    await storage.init_mysql()
    bot = Bot(token=BOT_TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
