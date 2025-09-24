# Enhanced Telegram Whisper Bot - Google Colab Version
# v4: Complete, unabridged script with prompting, language selection, and accuracy enhancements.

import os
import sys
import json
import re
import asyncio
import logging
import sqlite3
import hashlib
import time
import uuid
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

# --- Colab-specific Setup ---
# Apply nest_asyncio for Colab's event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ nest_asyncio applied for Colab compatibility")
except ImportError:
    print("⚠️ nest_asyncio not found, installing...")
    os.system("pip install -q nest_asyncio")
    import nest_asyncio
    nest_asyncio.apply()

# Install required packages if not available
print("📦 Checking and installing required packages...")
required_packages = [
    "pyrogram", "tgcrypto", "faster-whisper", "yt-dlp",
    "torch", "aiofiles", "aiohttp", "requests"
]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"  -> Installing {package}...")
        os.system(f"pip install -q {package}")
print("✅ All packages are ready.")

# --- Main Library Imports ---
import requests
import aiohttp
import aiofiles
from pyrogram import Client, filters, enums
from pyrogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, Message,
    CallbackQuery, User, Chat
)
from pyrogram.enums import ChatAction
from pyrogram.errors import FloodWait, UserNotParticipant

from faster_whisper import WhisperModel
import yt_dlp
import torch

# --- Configuration Management ---
@dataclass
class ColabBotConfig:
    """Configuration class for the bot."""
    api_id: int
    api_hash: str
    bot_token: str
    gist_url: str = ""
    database_path: str = "/content/bot_data.db"
    model_dir: str = "/content/models"
    download_dir: str = "/content/downloads"
    temp_dir: str = "/content/temp"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_duration: int = 3600  # 1 hour
    rate_limit_requests: int = 15
    rate_limit_window: int = 3600  # 1 hour
    admin_users: List[int] = None
    allowed_users: List[int] = None

    def __post_init__(self):
        if self.admin_users is None:
            self.admin_users = []
        if self.allowed_users is None:
            self.allowed_users = []

async def load_config_from_gist(gist_url: str) -> Dict[str, Any]:
    """Load configuration from a GitHub Gist URL."""
    try:
        if "gist.github.com" in gist_url and "/raw/" not in gist_url:
            gist_id = gist_url.split("/")[-1]
            raw_url = f"https://gist.githubusercontent.com/{gist_id}/raw/"
        else:
            raw_url = gist_url

        print(f"📡 Loading config from Gist: {raw_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(raw_url) as response:
                response.raise_for_status()
                text = await response.text()
                config_data = json.loads(text)
                print("✅ Config loaded successfully from Gist.")
                return config_data
    except Exception as e:
        print(f"❌ Error loading config from Gist: {e}")
        return {}

def get_manual_config() -> ColabBotConfig:
    """Get configuration through manual user input."""
    print("🔧 Manual Configuration Setup")
    print("=" * 50)
    api_id = input("Enter your Telegram API ID: ")
    api_hash = input("Enter your Telegram API Hash: ")
    bot_token = input("Enter your Bot Token: ")
    admin_input = input("Enter admin user IDs (comma-separated, optional): ")
    admin_users = [int(x.strip()) for x in admin_input.split(",") if x.strip().isdigit()]

    return ColabBotConfig(
        api_id=int(api_id),
        api_hash=api_hash,
        bot_token=bot_token,
        admin_users=admin_users
    )

async def initialize_config() -> ColabBotConfig:
    """Initialize configuration, prioritizing local file, then Gist, then manual."""
    local_config_path = "/content/colab_gist_config_template.json"

    if os.path.exists(local_config_path):
        print(f"✅ Found local config file: {local_config_path}")
        try:
            with open(local_config_path, 'r') as f:
                config_data = json.load(f)
                print("✅ Config loaded successfully from local file.")
                return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from local config file: {e}")

    gist_url = os.environ.get('CONFIG_GIST_URL') or input("Enter Config Gist URL (or press Enter for manual setup): ")
    if gist_url:
        try:
            config_data = await load_config_from_gist(gist_url)
            if config_data:
                config_data['gist_url'] = gist_url
                return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from gist: {e}")

    print("💡 No local file or Gist URL found, proceeding with manual setup.")
    return get_manual_config()

# --- Database Manager ---
class ColabDatabaseManager:
    """Manages the SQLite database for user data and logs."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        print(f"🗄️ Initializing database at: {db_path}")
        self.init_database()

    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY, username TEXT, first_name TEXT,
                        preferred_model TEXT DEFAULT 'large-v2',
                        usage_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS transcriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, file_hash TEXT,
                        model_used TEXT, language_detected TEXT, processing_time REAL,
                        file_size INTEGER, duration REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    );
                """)
            print("✅ Database initialized successfully.")
        except Exception as e:
            logging.error(f"Database initialization error: {e}", exc_info=True)
            raise

    def get_user(self, user_id: int) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logging.error(f"Error getting user {user_id}: {e}")
            return None

    def create_or_update_user(self, user: User):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (user_id, username, first_name, last_active)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username = excluded.username,
                        first_name = excluded.first_name,
                        last_active = excluded.last_active;
                """, (user.id, user.username, user.first_name))
        except Exception as e:
            logging.error(f"Error updating user {user.id}: {e}")

    def update_user_setting(self, user_id: int, key: str, value: Any):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"UPDATE users SET {key} = ? WHERE user_id = ?", (value, user_id))
        except Exception as e:
            logging.error(f"Error updating setting '{key}' for user {user_id}: {e}")

    def log_transcription(self, user_id: int, file_hash: str, model: str, lang: str, ptime: float, fsize: int, dur: float):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO transcriptions (user_id, file_hash, model_used, language_detected,
                                                processing_time, file_size, duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, file_hash, model, lang, ptime, fsize, dur))
                conn.execute("UPDATE users SET usage_count = usage_count + 1, last_active = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))
        except Exception as e:
            logging.error(f"Error logging transcription: {e}")

# --- Model Manager ---
class ColabModelManager:
    """Handles loading and managing faster-whisper models."""
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.current_model = None
        self.current_model_name = None
        self.model_info = {
            "tiny": "Fastest, low accuracy",
            "base": "Fast, balanced accuracy",
            "small": "Good balance of speed and accuracy",
            "medium": "High accuracy, slower",
            "large-v2": "Best accuracy, slowest",
        }

    async def load_model(self, model_name: str) -> WhisperModel:
        if model_name not in self.model_info:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"🤖 Loading model: {model_name}...")
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
            print("🔥 Using GPU (CUDA) acceleration.")
        else:
            device, compute_type = "cpu", "int8"
            print("🖥️ No GPU detected. Using CPU-optimized mode.")

        self.current_model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(self.model_dir))
        self.current_model_name = model_name
        print(f"✅ Model '{model_name}' loaded successfully on {device}.")
        return self.current_model

# --- Main Bot Class ---
class ColabWhisperBot:
    def __init__(self, config: ColabBotConfig):
        self.config = config
        self.db = ColabDatabaseManager(config.database_path)
        self.model_manager = ColabModelManager(config.model_dir)
        self.processing_status = {}
        self.app = Client("colab_whisper_bot", api_id=config.api_id, api_hash=config.api_hash, bot_token=config.bot_token, workdir="/content")
        self._setup_handlers()
        print("🤖 Bot class initialized successfully!")

    def _setup_handlers(self):
        self.app.on_message(filters.command("start"))(self.handle_start)
        self.app.on_message(filters.command("help"))(self.handle_help)
        self.app.on_message(filters.command("models"))(self.handle_models)
        self.app.on_message(filters.command("stats"))(self.handle_stats)
        self.app.on_message(filters.reply & filters.text & filters.regex(r"^/(transcribe|subtitle)"))(self.handle_media_processing)
        self.app.on_callback_query()(self.handle_callbacks)

    async def handle_start(self, client: Client, message: Message):
        self.db.create_or_update_user(message.from_user)
        welcome_text = (
            f"🎤 **Whisper Transcription Bot**\n\n"
            f"Hello {message.from_user.first_name}! To get started, simply reply to an audio or video file with `/transcribe`.\n\n"
            "Use `/help` to see all available features for improving accuracy."
        )
        await message.reply_text(welcome_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_help(self, client: Client, message: Message):
        help_text = (
            "📚 **Bot Help Guide**\n\n"
            "**How to Transcribe Audio**\n"
            "1. Reply to an audio, video, or voice message.\n"
            "2. Use one of the commands below:\n\n"
            "🔹 `/transcribe`\n"
            "   Gets the plain text transcription.\n\n"
            "🔹 `/subtitle`\n"
            "   Generates an `.srt` subtitle file.\n\n"
            "--- \n"
            "**💡 Pro Tips for Maximum Accuracy:**\n\n"
            "1. **Add a Prompt:** Guide the model by providing context, jargon, or names. This is the most powerful tool for accuracy.\n"
            "   *Example:* `/transcribe A discussion about adenocarcinoma and metformin.`\n\n"
            "2. **Set the Language:** Force a specific language if auto-detect struggles with accents.\n"
            "   *Example:* `/subtitle lang=en Your prompt here...`\n\n"
            "--- \n"
            "**Other Commands:**\n"
            "• `/models` - Choose a different transcription model (`large-v2` is best).\n"
            "• `/stats` - View your usage statistics."
        )
        await message.reply_text(help_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_models(self, client: Client, message: Message):
        buttons = [
            InlineKeyboardButton(f"{name} ({desc})", callback_data=f"select_model_{name}")
            for name, desc in self.model_manager.model_info.items()
        ]
        keyboard = InlineKeyboardMarkup([buttons[i:i+1] for i in range(0, len(buttons))]) # One button per row
        await message.reply_text(
            "**Select a Transcription Model:**\n\n"
            "- **`large-v2`**: Most accurate, recommended for best results.\n"
            "- **`small`/`medium`**: Good balance for faster processing.\n"
            "- **`base`/`tiny`**: Fastest, for non-critical tasks.",
            reply_markup=keyboard
        )

    async def handle_stats(self, client: Client, message: Message):
        user_data = self.db.get_user(message.from_user.id)
        if not user_data:
            return await message.reply_text("❌ No statistics available yet. Use the bot first!")
        stats_text = (
            f"📊 **Your Statistics**\n\n"
            f"👤 **User**: `{message.from_user.first_name}`\n"
            f"📈 **Total Transcriptions**: `{user_data.get('usage_count', 0)}`\n"
            f"🤖 **Preferred Model**: `{user_data.get('preferred_model', 'large-v2')}`"
        )
        await message.reply_text(stats_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_callbacks(self, client: Client, query: CallbackQuery):
        data = query.data
        if data.startswith("select_model_"):
            model_name = data.replace("select_model_", "")
            await query.answer(f"Loading {model_name}...")
            try:
                await self.model_manager.load_model(model_name)
                self.db.update_user_setting(query.from_user.id, 'preferred_model', model_name)
                await query.edit_message_text(f"✅ Model **{model_name}** is now active!", parse_mode=enums.ParseMode.MARKDOWN)
            except Exception as e:
                await query.edit_message_text(f"❌ Failed to load model: {e}", parse_mode=enums.ParseMode.MARKDOWN)
        else:
            await query.answer("Unknown action.", show_alert=True)

    async def handle_media_processing(self, client: Client, message: Message):
        # --- PARSE COMMAND, LANGUAGE, AND PROMPT ---
        text = message.text
        command_match = re.match(r"/(transcribe|subtitle)", text)
        command = command_match.group(0)
        remaining_text = text[len(command):].strip()
        
        lang_match = re.match(r"lang=(\w{2,3})\s*", remaining_text)
        language = None
        if lang_match:
            language = lang_match.group(1)
            initial_prompt = remaining_text[len(lang_match.group(0)):].strip()
        else:
            initial_prompt = remaining_text.strip()
        initial_prompt = initial_prompt or None

        status_message = await message.reply_text("⏳ Initializing...", parse_mode=enums.ParseMode.MARKDOWN)
        
        try:
            media_msg = message.reply_to_message
            media = media_msg.audio or media_msg.video or media_msg.voice or media_msg.video_note or media_msg.document
            if not media:
                raise ValueError("Please reply to a valid audio, video, or voice message.")

            await status_message.edit_text("📥 Downloading...")
            temp_path = await client.download_media(media, file_name=f"/content/temp/{uuid.uuid4()}")
            file_hash = hashlib.md5(Path(temp_path).read_bytes()).hexdigest()
            file_size = os.path.getsize(temp_path)
            
            await status_message.edit_text("🔄 Converting to WAV audio...")
            wav_path = await self._convert_to_wav(temp_path)

            if not self.model_manager.current_model:
                user_data = self.db.get_user(message.from_user.id)
                user_model = user_data.get('preferred_model', 'large-v2')
                await self.model_manager.load_model(user_model)

            await status_message.edit_text("🧠 Transcribing... (this may take a while with large models)")
            transcription, info = await self._transcribe_audio(wav_path, language=language, initial_prompt=initial_prompt)
            
            output_format = 'srt' if command == '/subtitle' else 'txt'
            output_content = self._format_output(transcription, info.duration, output_format)
            output_path = Path(f"/content/temp/transcript_{uuid.uuid4().hex[:6]}.{output_format}")
            output_path.write_text(output_content, encoding='utf-8')
            
            caption = (
                f"✅ **Transcription Complete**\n\n"
                f"**Language:** `{info.language}` (Confidence: {info.language_probability:.2f})\n"
                f"**Model:** `{self.model_manager.current_model_name}`\n"
                f"{'**Prompt:** Provided' if initial_prompt else ''}"
            ).strip()
            
            await client.send_document(message.chat.id, str(output_path), caption=caption, parse_mode=enums.ParseMode.MARKDOWN)
            await status_message.delete()
            
            self.db.log_transcription(message.from_user.id, file_hash, self.model_manager.current_model_name, info.language, info.duration, file_size, info.duration)
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        except Exception as e:
            logging.error(f"Processing error: {e}", exc_info=True)
            await status_message.edit_text(f"❌ **Error:** {str(e)}")

    async def _convert_to_wav(self, media_path: str) -> str:
        output_path = f"{media_path}.wav"
        cmd = ['ffmpeg', '-y', '-i', media_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path]
        process = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0: raise IOError(f"FFmpeg error: {stderr.decode()}")
        return output_path

    async def _transcribe_audio(self, audio_path: str, language: Optional[str], initial_prompt: Optional[str]):
        """Transcribes with accuracy-focused settings."""
        segments, info = self.model_manager.current_model.transcribe(
            audio_path,
            beam_size=10,
            vad_filter=True,
            language=language,
            initial_prompt=initial_prompt
        )
        full_text = " ".join(seg.text.strip() for seg in segments)
        return full_text, info

    def _format_output(self, text: str, duration: float, format_type: str) -> str:
        """Formats the transcription into plain text or SRT."""
        if format_type == 'srt':
            def to_srt_time(seconds: float) -> str:
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            return f"1\n00:00:00,000 --> {to_srt_time(duration)}\n{text}"
        return text

    async def run(self):
        """Starts the bot client and prints status."""
        print("🚀 Starting Bot Client...")
        await self.app.start()
        me = await self.app.get_me()
        print(f"✅ Bot started as @{me.username}!")
        print("🤖 Send /start to begin. Keep this Colab cell running.")

# --- Main Execution ---
async def main():
    """Initializes and runs the bot, handling the main lifecycle."""
    print("🔥 Enhanced Telegram Whisper Bot - Colab Edition")
    bot_instance = None
    try:
        config = await initialize_config()
        bot_instance = ColabWhisperBot(config)
        await bot_instance.run()
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Bot shutdown requested.")
    except Exception as e:
        print(f"❌ A critical startup error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if bot_instance and bot_instance.app.is_connected:
            print("👋 Stopping bot...")
            await bot_instance.app.stop()
            print("✅ Bot stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())

