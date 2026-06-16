"""Telegram bot — long polling через requests, без aiogram."""
import logging
import os
import re
import time
import threading
from datetime import datetime, timedelta

import markdown

import requests
from dotenv import load_dotenv
from openai import OpenAI

# --- Config ---
load_dotenv()

AUTHORIZED_USERS: set[int] = {
    int(uid) for uid in os.getenv("AUTHORIZED_USERS", "").split(",") if uid.strip()
}

BOT_TOKEN = os.getenv("BOT_TOKEN")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://100.74.219.85:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты полезный ассистент. Отвечай на языке пользователя.",
)

if not BOT_TOKEN:
    print("❌ Нет BOT_TOKEN в .env")
    exit(1)

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bot")


# --- State ---
chat_histories: dict[int, list[dict]] = {}



def get_history(chat_id: int) -> list[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return chat_histories[chat_id]


# --- LLM ---
llm_client = OpenAI(
    api_key=LLM_API_KEY or "no-key",
    base_url=LLM_BASE_URL,
)


def ask_llm(chat_id: int, user_text: str) -> str:
    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})

    logger.info(f"[{chat_id}] Sending to LLM ({len(history)} msgs)")

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=history,
            temperature=0.7,
            max_tokens=4096,
            timeout=120.0,
        )
        reply = response.choices[0].message.content or ""
        if not reply:
            msg = response.choices[0].message
            reply = getattr(msg, 'reasoning_content', None) or "(модель не вернула текст)"

        history.append({"role": "assistant", "content": reply})
        logger.info(f"[{chat_id}] LLM replied ({len(reply)} chars)")
    except Exception as e:
        logger.exception(f"LLM error for chat {chat_id}")
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            reply = "⚠️ Ошибка авторизации при обращении к LLM."
        elif "404" in msg or "Not Found" in msg:
            reply = f"⚠️ Модель '{LLM_MODEL}' не найдена на сервере."
        else:
            reply = f"⚠️ Ошибка LLM: {msg}"

    return reply


# --- Telegram API helpers ---
def tg_post(method: str, data: dict | None = None) -> dict:
    """POST to Telegram Bot API."""
    url = f"{TELEGRAM_API}/{method}"
    try:
        r = requests.post(url, json=data or {}, timeout=10)
        return r.json()
    except Exception as e:
        logger.exception(f"tg_post {method} failed: {e}")
        return {}


# Tags that Telegram HTML parser does NOT support — strip tag but keep text content
_UNTAG_TAGS = ["ul", "ol", "blockquote", "li"]
# Full removal (no replacement)
_REMOVE_TAGS = [r"<hr\s*/?>", r"<img[^>]*/?\s*>"]

_split_re = re.compile(r"\s{2,}")
def _split_text(text: str, limit: int) -> list[str]:
    """Split text into chunks of `limit` chars at word boundaries."""
    if len(text) <= limit:
        return [text]
    parts = []
    while len(text) > limit:
        # Try splitting at paragraph boundary (double+ whitespace)
        m = _split_re.search(text, 0, min(limit + 32, len(text)))
        if m and limit // 4 <= m.start() < limit + 16:
            end = m.start()
        else:
            # Fall back to last space before the limit
            end = text.rfind(" ", 0, limit)
            if end == -1 or end < limit // 4:
                end = limit
        parts.append(text[:end].rstrip())
        text = text[end:].lstrip()
    if text:
        parts.append(text)
    return parts


def _clean_telegram_html(html: str) -> str:
    """Remove/convert Telegram-incompatible tags from markdown output."""
    # Convert <hN>...</hN> → <b>...</b>
    for i in range(1, 7):
        html = re.sub(
            rf"<h{i}>(.*?)</h{i}>", r"<b>\1</b>", html, flags=re.DOTALL
        )

    # Strip unsupported tags but keep their inner text
    for tag in _UNTAG_TAGS:
        html = re.sub(rf"</{tag}>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(rf"<{tag}(?:[^>]*)?>", "", html, flags=re.IGNORECASE)

    # Remove class attributes from <pre> and <code> — Telegram needs bare tags
    html = re.sub(r'(<(pre|code))\s+class="[^"]*"', r'<\2', html)

    # Remove tags entirely (no replacement)
    for pat in _REMOVE_TAGS:
        html = re.sub(pat, "", html, flags=re.DOTALL | re.IGNORECASE)

    # Clean up paragraph wrappers — replace with newlines
    html = re.sub(r"<p>", "\n", html)
    html = re.sub(r"</p>", "\n", html)

    # Collapse multiple blank lines into at most one empty line
    html = re.sub(r"\n{3,}", "\n\n", html)

    return html.strip()


def send_message(chat_id: int, text: str):
    """Send a message to Telegram with Markdown→HTML conversion."""
    # Convert Markdown → HTML
    raw_html = markdown.markdown(
        text,
        extensions=["fenced_code", "codehilite", "tables"],
    )
    html_text = _clean_telegram_html(raw_html)

    # Telegram limit for HTML is 4096 chars (includes tags)
    MAX = 3800
    parts = _split_text(html_text, MAX)
    for idx, part in enumerate(parts):
        suffix = f"\n\n<i>(продолжение {idx+1}/{len(parts)})</i>" if len(parts) > 1 else ""
        tg_post("sendMessage", {
            "chat_id": chat_id,
            "text": part + suffix,
            "parse_mode": "HTML",
        })


# --- Long polling loop ---
offset = 0

logger.info("🚀 Бот запущен! Ожидание сообщений...")

while True:
    try:
        r = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": 15},
            timeout=20,
        )
        data = r.json()
    except Exception as e:
        logger.error(f"getUpdates failed: {e}")
        time.sleep(3)
        continue

    if not data.get("ok"):
        logger.warning(f"getUpdates returned error: {data}")
        time.sleep(1)
        continue

    for update in data["result"]:
        offset = max(offset, update["update_id"] + 1)

        msg = update.get("message", {})
        if not msg:
            continue

        chat_id = msg["chat"]["id"]

        # Check authorization
        if chat_id not in AUTHORIZED_USERS:
            logger.warning(f"Unauthorized access attempt from {chat_id}")
            continue

        text = (msg.get("text") or "").strip()
        if not text:
            continue

        # Ignore commands that start with /
        if text.startswith("/"):
            if text == "/new":
                chat_histories[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                send_message(chat_id, "✨ Новый чат начат.")
            elif text == "/start":
                send_message(chat_id, "👋 Привет! Я бот-прокси к LLM.\n\nПросто отправь сообщение — и я отвечу от имени модели.\n\n/new  — начать новый чат")
            continue

        logger.info(f"[{chat_id}] Got: {repr(text[:100])}")
        reply = ask_llm(chat_id, text)
        send_message(chat_id, reply)
