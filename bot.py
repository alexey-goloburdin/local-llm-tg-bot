"""Telegram bot — long polling через requests, без aiogram."""
import base64
import logging
import os
import re
import time
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
_model_supports_images: bool = True  # assume true, check once at startup


def _check_image_support_at_startup():
    """Check if the LLM model supports image inputs. Called once on startup."""
    global _model_supports_images
    try:
        llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                    {"type": "text", "text": "ok"}
                ]
            }],
            max_tokens=5,
            timeout=30.0,
        )
        logger.info(f"✅ Model '{LLM_MODEL}' supports images")
    except Exception as e:
        msg = str(e).lower()
        if "image" in msg and ("support" in msg or "vision" in msg):
            _model_supports_images = False
            logger.warning(f"⚠️ Model '{LLM_MODEL}' does NOT support images — will strip images from requests")
        else:
            # Other error — assume it supports images, handle per-request later
            logger.warning(f"Could not determine image support for '{LLM_MODEL}': {e}")



def get_history(chat_id: int) -> list[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return chat_histories[chat_id]


# --- Telegram file download ---
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def get_file_path(file_id: str) -> str | None:
    """Get the file path from Telegram for a given file_id."""
    r = tg_post("getFile", {"file_id": file_id})
    if r.get("ok") and "result" in r and isinstance(r["result"], dict):
        file_path = r["result"].get("file_path")
        if file_path:
            logger.debug(f"File path for {file_id}: {file_path}")
            return file_path
    logger.warning(f"getFile failed or no file_path for {file_id}: {r}")
    return None


def download_file(file_path: str) -> bytes | None:
    """Download a file from Telegram."""
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    try:
        logger.info(f"Downloading file: {url}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            size = len(r.content)
            logger.info(f"Downloaded {size} bytes")
            return r.content
        else:
            logger.error(f"Failed to download file: status={r.status_code}, body={r.text[:200]}")
    except Exception as e:
        logger.exception(f"Failed to download file {file_path}: {e}")
    return None


def get_image_from_message(msg: dict) -> bytes | None:
    """Extract image bytes from a Telegram message (photo or document)."""
    # Check photos first (Telegram sends them as an array, largest is last)
    if msg.get("photo"):
        photo = msg["photo"][-1]  # largest resolution
        file_id = photo["file_id"]
        logger.info(f"Photo found: file_id={file_id}, size={photo.get('width')},{photo.get('height')}")
        file_path = get_file_path(file_id)
        if file_path:
            return download_file(file_path)
    
    # Check documents for image MIME types
    doc = msg.get("document")
    if doc and doc.get("mime_type") in IMAGE_MIMES:
        file_id = doc["file_id"]
        logger.info(f"Image document found: mime={doc['mime_type']}, size={doc.get('file_size')}")
        file_path = get_file_path(file_id)
        if file_path:
            return download_file(file_path)
    
    # Check stickers (tgs/webp) — skip animated ones, handle static webp
    sticker = msg.get("sticker")
    if sticker and isinstance(sticker, dict):
        file_id = sticker["file_id"]
        logger.info(f"Sticker found: emoji={sticker.get('emoji')}")
        file_path = get_file_path(file_id)
        if file_path:
            return download_file(file_path)
    
    # Debug: log message types for troubleshooting
    keys = list(msg.keys())
    logger.info(f"No image found in msg. Keys: {keys}")
    return None


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 data URL."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Guess MIME type from magic bytes
    mime = "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":  # JPEG
        mime = "image/jpeg"
    elif b"WEBP" in image_bytes[8:12]:
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"


# --- LLM ---
llm_client = OpenAI(
    api_key=LLM_API_KEY or "no-key",
    base_url=LLM_BASE_URL,
)


def ask_llm(chat_id: int, user_text: str | None, image_bytes: bytes | None) -> str:
    global _model_supports_images
    history = get_history(chat_id)
    
    if image_bytes and _model_supports_images:
        b64 = image_to_base64(image_bytes)
        content_parts = [{"type": "image_url", "image_url": {"url": b64}}]
        if user_text:
            content_parts.append({"type": "text", "text": user_text})
        else:
            content_parts.append({"type": "text", "text": "Опиши что изображено на фото."})
        history.append({"role": "user", "content": content_parts})
    else:
        if image_bytes and not _model_supports_images:
            logger.info(f"[chat={chat_id}] Skipping image — model doesn't support vision")
        logger.info(f"[chat={chat_id}] Sending text only: {repr((user_text or '')[:100])}")
        history.append({"role": "user", "content": user_text or ""})

    logger.info(f"[{chat_id}] Sending to LLM ({len(history)} msgs)")

    try:
        logger.info(f"[chat={chat_id}] Calling LLM API...")
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=history,
            temperature=0.7,
            max_tokens=4096,
            timeout=120.0,
        )
        logger.info(f"[chat={chat_id}] LLM API call succeeded")
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

_check_image_support_at_startup()

logger.info("🚀 Бот запущен! Ожидание сообщений...")

while True:
    try:
        logger.info(f"[GETUPDATES] offset={offset}")
        r = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": 15, "allowed_updates": ["message"]},
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
        old_offset = offset
        offset = max(offset, update["update_id"] + 1)
        logger.info(f"[UPDATE] update_id={update['update_id']} offset:{old_offset}->{offset}")

        msg = update.get("message", {})
        if not msg:
            continue

        chat_id = msg["chat"]["id"]

        # Check authorization
        if chat_id not in AUTHORIZED_USERS:
            logger.warning(f"Unauthorized access attempt from {chat_id}")
            continue

        # Check for images/photos/documents first
        image_bytes = get_image_from_message(msg)

        # Debug: log all message fields for troubleshooting
        logger.info(f"[MSG] keys={list(msg.keys())} chat_id={chat_id} has_image={bool(image_bytes)}")
        
        # Extract text from message (caption for photos/docs, or message.text)
        text = None
        if msg.get("text"):
            text = msg["text"]
        elif msg.get("photo") and msg["photo"][-1].get("caption"):
            text = msg["photo"][-1]["caption"]
        elif msg.get("document") and msg["document"].get("caption"):
            text = msg["document"]["caption"]
        if text:
            text = text.strip()

        # If no text and no image, skip (empty/unknown message type)
        if not text and not image_bytes:
            continue

        # Ignore commands that start with /
        if text and text.startswith("/"):
            if text == "/new":
                chat_histories[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                send_message(chat_id, "✨ Новый чат начат.")
            elif text == "/start":
                send_message(chat_id, "👋 Привет! Я бот-прокси к LLM.\n\nПросто отправь сообщение — и я отвечу от имени модели.\n\n/new  — начать новый чат")
            continue

        logger.info(f"[{chat_id}] Got: {repr((text or '')[:100])} {'[image]' if image_bytes else ''}")
        try:
            reply = ask_llm(chat_id, text, image_bytes)
            send_message(chat_id, reply)
        except Exception as e:
            logger.exception(f"Failed to send reply for chat {chat_id}: {e}")
