# Enhanced Telegram Whisper Bot - Google Colab Version
# Optimized for Google Colab with nested asyncio, prompting, and local/gist config loading

import os
import sys
import json
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
    log_level: str = "INFO"
    enable_translation: bool = True
    enable_voice_messages: bool = True
    auto_delete_files: bool = True
    delete_timeout: int = 300  # 5 minutes

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
    gist_url = input("Enter Gist URL for config updates (optional): ") or ""
    admin_input = input("Enter admin user IDs (comma-separated, optional): ")
    admin_users = [int(x.strip()) for x in admin_input.split(",") if x.strip().isdigit()]

    return ColabBotConfig(
        api_id=int(api_id),
        api_hash=api_hash,
        bot_token=bot_token,
        gist_url=gist_url,
        admin_users=admin_users
    )

async def initialize_config() -> ColabBotConfig:
    """Initialize configuration, prioritizing local file, then Gist, then manual."""
    local_config_path = "/content/colab_gist_config_template.json"

    # 1. Try to load from local file first
    if os.path.exists(local_config_path):
        print(f"✅ Found local config file: {local_config_path}")
        try:
            with open(local_config_path, 'r') as f:
                config_data = json.load(f)
                print("✅ Config loaded successfully from local file.")
                return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from local config file: {e}")

    # 2. Fallback to Gist URL
    gist_url = os.environ.get('CONFIG_GIST_URL') or input("Enter Config Gist URL (or press Enter for manual setup): ")
    if gist_url:
        try:
            config_data = await load_config_from_gist(gist_url)
            if config_data:
                config_data['gist_url'] = gist_url
                return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from gist: {e}")

    # 3. Fallback to manual configuration
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
                        language_code TEXT DEFAULT 'auto', preferred_model TEXT DEFAULT 'base',
                        translate_to TEXT DEFAULT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP, is_premium BOOLEAN DEFAULT FALSE,
                        usage_count INTEGER DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS transcriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, file_hash TEXT,
                        model_used TEXT, language_detected TEXT, processing_time REAL,
                        file_size INTEGER, duration REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    );
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        user_id INTEGER, timestamp TIMESTAMP, request_count INTEGER DEFAULT 1,
                        PRIMARY KEY (user_id, timestamp)
                    );
                    CREATE INDEX IF NOT EXISTS idx_rate_limits_user_time ON rate_limits (user_id, timestamp);
                """)
            print("✅ Database initialized successfully.")
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
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

    def check_rate_limit(self, user_id: int, max_requests: int, window_seconds: int) -> bool:
        try:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT SUM(request_count) FROM rate_limits WHERE user_id = ? AND timestamp > ?",
                    (user_id, cutoff_time)
                )
                current_requests = (cursor.fetchone() or [0])[0] or 0
                if current_requests >= max_requests:
                    return False

                now = datetime.now().replace(second=0, microsecond=0)
                conn.execute("""
                    INSERT INTO rate_limits (user_id, timestamp, request_count) VALUES (?, ?, 1)
                    ON CONFLICT(user_id, timestamp) DO UPDATE SET request_count = request_count + 1;
                """, (user_id, now))
                return True
        except Exception as e:
            logging.error(f"Rate limit check error: {e}")
            return True

    def log_transcription(self, user_id: int, file_hash: str, model: str, lang: str, ptime: float, fsize: int, dur: float):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO transcriptions (user_id, file_hash, model_used, language_detected,
                                                processing_time, file_size, duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, file_hash, model, lang, ptime, fsize, dur))
                conn.execute("UPDATE users SET usage_count = usage_count + 1 WHERE user_id = ?", (user_id,))
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
            "tiny": {"vram": "~0.8GB", "speed": "~32x", "recommended": "Speed"},
            "base": {"vram": "~1GB", "speed": "~16x", "recommended": "Balanced"},
            "small": {"vram": "~1.4GB", "speed": "~6x", "recommended": "Quality"},
            "medium": {"vram": "~2.7GB", "speed": "~2x", "recommended": "High Quality"},
            "large-v2": {"vram": "~4.3GB", "speed": "~1x", "recommended": "Best Quality"},
        }

    async def load_model(self, model_name: str) -> WhisperModel:
        if model_name not in self.model_info:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"🤖 Loading model: {model_name}...")
        # THIS SECTION HANDLES CPU/GPU FALLBACK AUTOMATICALLY
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
            print("🔥 Using GPU (CUDA) acceleration.")
        else:
            device, compute_type = "cpu", "int8"
            print("🖥️ No GPU detected. Using CPU-optimized mode.")

        await self._cleanup_old_models()
        self.current_model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(self.model_dir))
        self.current_model_name = model_name
        print(f"✅ Model '{model_name}' loaded successfully on {device}.")
        return self.current_model

    async def _cleanup_old_models(self):
        try:
            model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir() and 'models--' in d.name]
            if len(model_dirs) > 1:
                print("🧹 Cleaning up old models to save space...")
                model_dirs.sort(key=lambda x: x.stat().st_mtime)
                for old_dir in model_dirs[:-1]:
                    shutil.rmtree(old_dir)
                    print(f"  -> Removed: {old_dir.name}")
        except Exception as e:
            logging.warning(f"Model cleanup warning: {e}")

# --- Main Bot Class ---
class ColabWhisperBot:
    def __init__(self, config: ColabBotConfig):
        self.config = config
        self.db = ColabDatabaseManager(config.database_path)
        self.model_manager = ColabModelManager(config.model_dir)
        self.processing_status = {}

        for directory in [config.model_dir, config.download_dir, config.temp_dir]:
            Path(directory).mkdir(exist_ok=True, parents=True)

        self.app = Client("colab_whisper_bot", api_id=config.api_id, api_hash=config.api_hash,
                          bot_token=config.bot_token, workdir="/content")
        self._setup_handlers()
        print("🤖 Bot class initialized successfully!")

    def _setup_handlers(self):
        self.app.on_message(filters.command("start"))(self.handle_start)
        self.app.on_message(filters.command("help"))(self.handle_help)
        self.app.on_message(filters.command("status"))(self.handle_status)
        self.app.on_message(filters.command("models"))(self.handle_models)
        self.app.on_message(filters.command("stats"))(self.handle_stats)
        self.app.on_message(filters.reply & (filters.audio | filters.video | filters.voice | filters.video_note | filters.document | filters.text))(self.handle_media_processing)
        self.app.on_callback_query()(self.handle_callbacks)

    async def handle_start(self, client: Client, message: Message):
        self.db.create_or_update_user(message.from_user)
        welcome_text = (
            f"🎤 **Enhanced Whisper Bot (Colab Edition)**\n\n"
            f"Hello {message.from_user.first_name}! I'm running on Google Colab with advanced transcription capabilities.\n\n"
            "**✨ New Feature: Prompting!**\n"
            "Improve accuracy for medical or technical terms by providing a prompt. Example: `/transcribe A discussion about adenocarcinoma.`\n\n"
            "Use `/help` for a full list of commands!"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🤖 Models", callback_data="models"), InlineKeyboardButton("📚 Help", callback_data="help")],
            [InlineKeyboardButton("📊 My Stats", callback_data="stats")]
        ])
        await message.reply_text(welcome_text, parse_mode=enums.ParseMode.MARKDOWN, reply_markup=keyboard)

    async def handle_help(self, client: Client, message: Message):
        help_text = (
            "📚 **Colab Whisper Bot - Help Guide**\n\n"
            "**🎵 Processing Commands (reply to a media file):**\n"
            "• `/transcribe` - Get the audio as plain text.\n"
            "• `/subtitle` - Generate an `.srt` subtitle file.\n\n"
            "**💡 Improve Accuracy with Prompts!**\n"
            "To correctly transcribe specific jargon, names, or technical terms, add them after the command.\n"
            "  - *Example:* `/transcribe A meeting about Kubernetes and Docker.`\n"
            "  - *Medical:* `/subtitle The patient has paroxysmal atrial fibrillation.`\n\n"
            "**⚙️ Bot Commands:**\n"
            "• `/models` - View and select a transcription model.\n"
            "• `/stats` - View your personal usage statistics.\n"
            "• `/status` - Check the bot's current processing status."
        )
        await message.reply_text(help_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_status(self, client: Client, message: Message):
        status = self.processing_status.get(message.chat.id)
        if status:
            elapsed = time.time() - status.get('start_time', time.time())
            status_text = (
                f"📊 **Current Task Status**\n\n"
                f"🔄 **Process**: {status.get('message', 'Working...')}\n"
                f"📈 **Progress**: {status.get('progress', 0):.1f}%\n"
                f"⏱️ **Elapsed Time**: {elapsed:.1f}s"
            )
        else:
            status_text = (
                f"✅ **Bot Status**\n\n"
                f"🤖 **Model Loaded**: `{self.model_manager.current_model_name or 'None'}`\n"
                f"💾 **Device**: `{'GPU' if torch.cuda.is_available() else 'CPU'}`\n"
                f"🔄 **Active Processes**: None\n\n"
                "I'm ready to process your media!"
            )
        await message.reply_text(status_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_models(self, client: Client, message: Message):
        models_text = "🤖 **Available Models (Colab Optimized)**\n\n"
        for name, info in self.model_manager.model_info.items():
            emoji = "⚡" if name in ["tiny", "base"] else "🎯"
            models_text += f"{emoji} **{name}**\n"
            models_text += f"├ Memory: {info['vram']}\n"
            models_text += f"├ Speed: {info['speed']}\n"
            models_text += f"└ Best for: {info['recommended']}\n\n"
        models_text += "💡 Use `small` or `medium` for better accuracy, especially for important content."

        buttons = [
            InlineKeyboardButton(f"{'⚡' if n in ['tiny', 'base'] else '🎯'} {n}", callback_data=f"select_model_{n}")
            for n in self.model_manager.model_info.keys()
        ]
        keyboard = InlineKeyboardMarkup([buttons[:3], buttons[3:]])
        await message.reply_text(models_text, parse_mode=enums.ParseMode.MARKDOWN, reply_markup=keyboard)

    async def handle_stats(self, client: Client, message: Message):
        user_data = self.db.get_user(message.from_user.id)
        if not user_data:
            return await message.reply_text("❌ No statistics available yet. Use the bot first!")

        stats_text = (
            f"📊 **Your Statistics**\n\n"
            f"👤 **User ID**: `{user_data['user_id']}`\n"
            f"📈 **Total Transcriptions**: `{user_data['usage_count']}`\n"
            f"🗓️ **Member Since**: `{user_data['created_at'][:10]}`\n"
            f"🤖 **Preferred Model**: `{user_data['preferred_model']}`"
        )
        await message.reply_text(stats_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_media_processing(self, client: Client, message: Message):
        if not message.reply_to_message: return

        user_id = message.from_user.id
        chat_id = message.chat.id
        
        # --- NEW: PARSE COMMAND AND PROMPT ---
        command_parts = message.text.split(maxsplit=1)
        command = command_parts[0].lower()
        initial_prompt = command_parts[1] if len(command_parts) > 1 else None
        
        if not (command.startswith('/transcribe') or command.startswith('/subtitle')):
            return

        if not self.db.check_rate_limit(user_id, self.config.rate_limit_requests, self.config.rate_limit_window):
            return await message.reply_text("⏱️ Rate limit exceeded. Please wait a bit before making new requests.")

        if chat_id in self.processing_status:
            return await message.reply_text("⚠️ Please wait for the current process to finish before starting a new one.")

        status_message = await message.reply_text("⏳ **Initializing...**", parse_mode=enums.ParseMode.MARKDOWN)
        self.processing_status[chat_id] = {"start_time": time.time()}

        try:
            # Download
            await self._update_status(status_message, "📥 Downloading media...", 10)
            media_info = await self._download_media(client, message.reply_to_message)
            if not media_info: raise ValueError("Failed to download or invalid file.")
            file_path, file_hash, file_size = media_info

            # Convert
            await self._update_status(status_message, "🔄 Converting to audio...", 30)
            audio_path = await self._convert_to_wav(file_path)
            if not audio_path: raise ValueError("Audio conversion failed.")

            # Load model
            if not self.model_manager.current_model:
                await self._update_status(status_message, "🤖 Loading model...", 40)
                user_data = self.db.get_user(user_id)
                await self.model_manager.load_model(user_data['preferred_model'] if user_data else 'base')

            # Transcribe (with prompt)
            await self._update_status(status_message, "👂 Transcribing...", 60)
            transcription, lang, ptime, duration = await self._transcribe_audio(audio_path, initial_prompt=initial_prompt)
            if transcription is None: raise ValueError("Transcription process failed.")
            
            # Prepare output
            await self._update_status(status_message, "📄 Generating output...", 95)
            output_content = self._format_output(transcription, duration, '/subtitle' in command)
            output_filename = f"transcript_{uuid.uuid4().hex[:6]}.{'srt' if '/subtitle' in command else 'txt'}"
            output_path = Path(self.config.temp_dir) / output_filename
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(output_content)

            # Send results
            caption = (
                f"✅ **Processing Complete!**\n\n"
                f"🗣️ **Language**: `{lang}`\n"
                f"⏱️ **Time**: `{ptime:.1f}s`\n"
                f"🤖 **Model**: `{self.model_manager.current_model_name}`\n"
                f"{'📝 **Prompt Used**' if initial_prompt else ''}"
            )
            await client.send_document(chat_id, document=str(output_path), caption=caption,
                                       parse_mode=enums.ParseMode.MARKDOWN,
                                       reply_to_message_id=message.reply_to_message.id)
            await status_message.delete()

            self.db.log_transcription(user_id, file_hash, self.model_manager.current_model_name, lang, ptime, file_size, duration)
            for p in [file_path, audio_path, output_path]:
                if p and Path(p).exists(): Path(p).unlink()

        except Exception as e:
            logging.error(f"Processing error: {e}", exc_info=True)
            await status_message.edit_text(f"❌ **Error:** {str(e)}", parse_mode=enums.ParseMode.MARKDOWN)
        finally:
            if chat_id in self.processing_status:
                del self.processing_status[chat_id]

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
        elif data == "help":
            await query.answer()
            await self.handle_help(client, query.message)
        elif data == "models":
            await query.answer()
            await self.handle_models(client, query.message)
        elif data == "stats":
            await query.answer()
            await self.handle_stats(client, query.message)

    async def _update_status(self, msg: Message, text: str, progress: int):
        self.processing_status[msg.chat.id].update({"message": text, "progress": progress})
        try:
            await msg.edit_text(f"**{text}** ({progress}%)", parse_mode=enums.ParseMode.MARKDOWN)
        except Exception:
            pass

    async def _download_media(self, client: Client, message: Message):
        media = message.audio or message.video or message.voice or message.video_note or message.document
        if not media:
            if message.text and ("youtube.com" in message.text or "youtu.be" in message.text):
                return await self._download_youtube(message.text.strip())
            return None

        if hasattr(media, 'file_size') and media.file_size > self.config.max_file_size:
            raise ValueError(f"File is too large ({media.file_size / 1024**2:.1f}MB). Max is {self.config.max_file_size / 1024**2}MB.")

        temp_path = await client.download_media(message, file_name=f"{self.config.temp_dir}/{uuid.uuid4()}")
        file_hash = hashlib.md5(Path(temp_path).read_bytes()).hexdigest()
        file_size = os.path.getsize(temp_path)
        return temp_path, file_hash, file_size

    async def _download_youtube(self, url: str):
        temp_id = uuid.uuid4().hex
        output_path = f"{self.config.temp_dir}/{temp_id}.%(ext)s"
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 'outtmpl': output_path,
            'quiet': True, 'no_warnings': True, 'noplaylist': True,
            'max_filesize': self.config.max_file_size
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            file_size = os.path.getsize(downloaded_path)
            file_hash = hashlib.md5(Path(downloaded_path).read_bytes()).hexdigest()
            return downloaded_path, file_hash, file_size
        return None

    async def _convert_to_wav(self, media_path: str):
        input_path = Path(media_path)
        output_path = input_path.with_suffix('.wav')
        cmd = ['ffmpeg', '-y', '-i', str(input_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(output_path)]
        process = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise IOError(f"FFmpeg error: {stderr.decode()}")
        input_path.unlink()
        return str(output_path)

    # --- NEW: ACCEPTS AN INITIAL_PROMPT ---
    async def _transcribe_audio(self, audio_path: str, initial_prompt: Optional[str] = None):
        """Transcribes audio using the loaded model, with optional prompting."""
        start_time = time.time()
        duration = await self._get_audio_duration(audio_path)
        
        segments, info = self.model_manager.current_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True, # Voice Activity Detection is enabled here
            initial_prompt=initial_prompt # The new prompt is used here
        )
        
        transcription = " ".join(seg.text.strip() for seg in segments)
        processing_time = time.time() - start_time
        return transcription, info.language, processing_time, duration

    async def _get_audio_duration(self, audio_path: str) -> float:
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await process.communicate()
            return float(stdout.decode().strip())
        except Exception:
            return 0.0

    def _format_output(self, text: str, duration: float, is_srt: bool) -> str:
        if is_srt:
            return f"1\n00:00:00,000 --> {self._seconds_to_srt_time(duration)}\n{text}"
        return text

    def _seconds_to_srt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

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
    print("=" * 60)
    bot_instance = None
    try:
        # Initialize configuration and bot instance
        config = await initialize_config()
        bot_instance = ColabWhisperBot(config)

        # Start the bot
        await bot_instance.run()

        # Keep the script running indefinitely until interrupted
        await asyncio.Event().wait()

    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Bot shutdown requested.")
    except Exception as e:
        print(f"❌ A critical startup error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot_instance and bot_instance.app.is_connected:
            print("👋 Stopping bot...")
            await bot_instance.app.stop()
            print("✅ Bot stopped.")

if __name__ == "__main__":
    try:
        import google.colab
        print("✅ Google Colab environment detected.")
    except ImportError:
        print("⚠️ Not running in a standard Google Colab environment.")
    
    # Run the main asynchronous function
    asyncio.run(main())

