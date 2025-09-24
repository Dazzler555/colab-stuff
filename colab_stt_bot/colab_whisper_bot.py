# Enhanced Telegram Whisper Bot - Google Colab Version
# v6: Final version with anti-repetition filters and all previous fixes.

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
from typing import Optional, Dict, List, Tuple, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict

# --- Colab-specific Setup ---
# Apply nest_asyncio for Colab's event loop, a cornerstone for notebook compatibility.
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
    CallbackQuery, User
)
from pyrogram.errors import FloodWait

from faster_whisper import WhisperModel
import yt_dlp
import torch

# --- Configuration Management ---
@dataclass
class ColabBotConfig:
    """Comprehensive configuration class for the bot."""
    api_id: int
    api_hash: str
    bot_token: str
    gist_url: str = ""
    database_path: str = "/content/bot_data.db"
    model_dir: str = "/content/models"
    download_dir: str = "/content/downloads"
    temp_dir: str = "/content/temp"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_duration: int = 7200  # 2 hours
    admin_users: List[int] = None

    def __post_init__(self):
        if self.admin_users is None:
            self.admin_users = []
        # Create necessary directories
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        Path(self.download_dir).mkdir(exist_ok=True, parents=True)
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

async def load_config_from_gist(gist_url: str) -> Dict[str, Any]:
    """Load configuration from a GitHub Gist URL."""
    try:
        # Streamlined Gist URL parsing
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
            return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from local config file: {e}")

    gist_url = os.environ.get('CONFIG_GIST_URL') or input("Enter Config Gist URL (or press Enter for manual setup): ")
    if gist_url:
        config_data = await load_config_from_gist(gist_url)
        if config_data:
            config_data['gist_url'] = gist_url
            return ColabBotConfig(**config_data)

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
                        preferred_model TEXT DEFAULT 'large-v3',
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
    """Handles loading and managing faster-whisper models, including custom ones."""
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.current_model: Optional[WhisperModel] = None
        self.current_model_name: Optional[str] = None
        self.model_info = {
            "tiny": "Fastest, low accuracy",
            "base": "Fast, balanced accuracy",
            "small": "Good balance of speed and accuracy",
            "medium": "High accuracy, slower",
            "large-v3": "Best accuracy, default",
        }

    async def load_model(self, model_name_or_path: str) -> WhisperModel:
        print(f"🤖 Loading model: {model_name_or_path}...")
        
        # Determine device and compute type. This is a robust check for hardware acceleration.
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
            print("🔥 Using GPU (CUDA) acceleration.")
        else:
            device, compute_type = "cpu", "int8"
            print("🖥️ No GPU detected. Using CPU-optimized mode (int8).")

        # Handle custom model paths
        if os.path.isdir(model_name_or_path):
            print(f"📁 Loading custom model from path: {model_name_or_path}")
            model_id = Path(model_name_or_path).name
        else:
            if model_name_or_path not in self.model_info:
                raise ValueError(f"Unknown model: {model_name_or_path}. Use /models to see options.")
            model_id = model_name_or_path

        self.current_model = WhisperModel(model_id, device=device, compute_type=compute_type, download_root=str(self.model_dir))
        self.current_model_name = model_id
        print(f"✅ Model '{model_id}' loaded successfully on {device}.")
        return self.current_model

# --- Main Bot Class ---
YOUTUBE_REGEX = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})"

class ColabWhisperBot:
    def __init__(self, config: ColabBotConfig):
        self.config = config
        self.db = ColabDatabaseManager(config.database_path)
        self.model_manager = ColabModelManager(config.model_dir)
        self.app = Client("colab_whisper_bot", api_id=config.api_id, api_hash=config.api_hash, bot_token=config.bot_token, workdir="/content")
        self._setup_handlers()
        print("🤖 Bot class initialized successfully!")

    def _setup_handlers(self):
        """A streamlined method to register all message and callback handlers."""
        self.app.on_message(filters.command("start"))(self.handle_start)
        self.app.on_message(filters.command("help"))(self.handle_help)
        self.app.on_message(filters.command("models"))(self.handle_models)
        self.app.on_message(filters.command("stats"))(self.handle_stats)
        self.app.on_message(filters.command("load_model") & filters.private)(self.handle_load_custom_model)
        
        # Media handlers
        self.app.on_message(filters.reply & filters.text & filters.regex(r"^/(transcribe|subtitle)"))(self.handle_media_command)
        self.app.on_message(filters.regex(YOUTUBE_REGEX) & filters.private)(self.handle_youtube_link)
        
        self.app.on_callback_query()(self.handle_callbacks)

    # --- Command Handlers ---
    async def handle_start(self, client: Client, message: Message):
        self.db.create_or_update_user(message.from_user)
        welcome_text = (
            f"🎤 **Whisper Transcription Bot**\n\n"
            f"Hello {message.from_user.first_name}! To get started:\n"
            "1. Reply to an audio/video file with `/transcribe`.\n"
            "2. Send me a YouTube link.\n\n"
            "Use `/help` to see all available features for improving accuracy."
        )
        await message.reply_text(welcome_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_help(self, client: Client, message: Message):
        help_text = (
            "📚 **Bot Help Guide**\n\n"
            "**How to Transcribe**\n"
            "1. **Reply to an audio/video file** or **send a YouTube link**.\n"
            "2. Use one of the commands below (or no command for YouTube links):\n\n"
            "🔹 `/transcribe` - Gets the plain text transcription.\n"
            "🔹 `/subtitle` - Generates an `.srt` subtitle file.\n\n"
            "--- \n"
            "**💡 Pro Tips for Maximum Accuracy**\n\n"
            "1. **Add a Prompt:** This is the most powerful tool. Guide the model with context, jargon, or names.\n"
            "   *Example:* `/transcribe A discussion about adenocarcinoma and metformin.`\n\n"
            "2. **Set the Language:** Force a language if auto-detect struggles.\n"
            "   *Example:* `/subtitle lang=en Your prompt here...`\n\n"
            "**Improving Accuracy for Accents (e.g., Indian English):**\n"
            "Provide context and proper nouns common in the audio. This helps the model anchor its transcription.\n"
            "*Example:* `/transcribe lang=en A conversation about the Indian Premier League (IPL) and players like Virat Kohli.`\n\n"
            "--- \n"
            "**Other Commands**\n"
            "• `/models` - Choose a different transcription model (`large-v3` is best).\n"
            "• `/stats` - View your usage statistics."
        )
        await message.reply_text(help_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_models(self, client: Client, message: Message):
        buttons = [InlineKeyboardButton(f"{name} ({desc})", callback_data=f"select_model_{name}") for name, desc in self.model_manager.model_info.items()]
        keyboard = InlineKeyboardMarkup([buttons[i:i+1] for i in range(0, len(buttons))]) # One button per row
        await message.reply_text(
            "**Select a Transcription Model:**\n\n"
            "- **`large-v3`**: Most accurate, recommended for best results.\n"
            "- **`small`/`medium`**: Good balance for faster processing.\n"
            "- **`tiny`/`base`**: Fastest, for non-critical tasks.",
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
            f"🤖 **Preferred Model**: `{user_data.get('preferred_model', 'large-v3')}`"
        )
        await message.reply_text(stats_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_load_custom_model(self, client: Client, message: Message):
        if message.from_user.id not in self.config.admin_users:
            return await message.reply_text("🚫 This is an admin-only command.")
        
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            return await message.reply_text("Usage: `/load_model /path/to/your/model_directory`")
        
        model_path = parts[1].strip()
        if not os.path.isdir(model_path):
            return await message.reply_text(f"❌ Path not found or is not a directory: `{model_path}`")

        status_msg = await message.reply_text(f"⏳ Loading custom model from `{model_path}`...")
        try:
            await self.model_manager.load_model(model_path)
            await status_msg.edit_text(f"✅ Custom model **{Path(model_path).name}** loaded successfully!")
        except Exception as e:
            await status_msg.edit_text(f"❌ Failed to load custom model: {e}")

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

    # --- Media Processing Logic ---
    async def handle_youtube_link(self, client: Client, message: Message):
        """Handles YouTube link messages."""
        status_msg = await message.reply_text("🔗 Processing YouTube link...", parse_mode=enums.ParseMode.MARKDOWN)
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.config.temp_dir, '%(id)s.%(ext)s'),
                'noplaylist': True,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await status_msg.edit_text("📥 Downloading audio from YouTube...")
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(message.text, download=True))
                downloaded_path = ydl.prepare_filename(info)

            # Process the downloaded file with default settings
            await self._process_media_file(message, downloaded_path, status_msg, command='/transcribe', initial_prompt=None, language=None)
        
        except Exception as e:
            logging.error(f"YouTube processing error: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ **Error processing YouTube link:** {str(e)}")

    async def handle_media_command(self, client: Client, message: Message):
        """Handles /transcribe and /subtitle commands on replied-to media."""
        # --- PARSE COMMAND, LANGUAGE, AND PROMPT ---
        text = message.text
        command_match = re.match(r"/(transcribe|subtitle)", text)
        command = command_match.group(0)
        remaining_text = text[len(command):].strip()
        
        lang_match = re.match(r"lang=(\w{2,3})\s*", remaining_text)
        language = lang_match.group(1) if lang_match else None
        initial_prompt = remaining_text[len(lang_match.group(0)):].strip() if lang_match else remaining_text
        initial_prompt = initial_prompt or None

        status_message = await message.reply_text("⏳ Initializing...", parse_mode=enums.ParseMode.MARKDOWN)
        
        try:
            media_msg = message.reply_to_message
            media = media_msg.audio or media_msg.video or media_msg.voice or media_msg.video_note or media_msg.document
            if not media:
                raise ValueError("Please reply to a valid audio or video file.")

            await status_message.edit_text("📥 Downloading file...")
            temp_path = await client.download_media(media, file_name=f"{self.config.temp_dir}/{uuid.uuid4()}")
            
            await self._process_media_file(message, temp_path, status_message, command, initial_prompt, language)

        except Exception as e:
            logging.error(f"Media processing error: {e}", exc_info=True)
            await status_message.edit_text(f"❌ **Error:** {str(e)}")

    async def _process_media_file(self, message: Message, file_path: str, status_msg: Message, command: str, initial_prompt: Optional[str], language: Optional[str]):
        """Core logic for processing a local media file."""
        start_time = time.time()
        temp_path = Path(file_path)
        try:
            file_hash = hashlib.md5(temp_path.read_bytes()).hexdigest()
            file_size = temp_path.stat().st_size
            
            await status_msg.edit_text("🔄 Converting to WAV audio...")
            wav_path = await self._convert_to_wav(str(temp_path))

            if not self.model_manager.current_model:
                user_data = self.db.get_user(message.from_user.id)
                user_model = user_data.get('preferred_model', 'large-v3')
                await self.model_manager.load_model(user_model)

            await status_msg.edit_text("🧠 Transcribing... (this may take a while)")
            
            # --- REAL-TIME TRANSCRIPTION PROGRESS ---
            full_text = ""
            last_update = 0
            async for segment, info in self._transcribe_audio_realtime(wav_path, language=language, initial_prompt=initial_prompt):
                full_text += segment.text
                if time.time() - last_update > 5:  # Update every 5 seconds
                    progress_text = f"🧠 **Transcribing...**\n\n`{full_text[-200:]}...`"
                    try:
                        await status_msg.edit_text(progress_text)
                        last_update = time.time()
                    except FloodWait as e:
                        await asyncio.sleep(e.value)
            
            output_format = 'srt' if command == '/subtitle' else 'txt'
            output_content = self._format_output(full_text.strip(), info, output_format)
            output_path = Path(f"{self.config.temp_dir}/transcript_{uuid.uuid4().hex[:6]}.{output_format}")
            output_path.write_text(output_content, encoding='utf-8')
            
            caption = (
                f"✅ **Transcription Complete**\n\n"
                f"**Language:** `{info.language}` (Confidence: {info.language_probability:.2f})\n"
                f"**Model:** `{self.model_manager.current_model_name}`\n"
                f"**Duration:** `{timedelta(seconds=int(info.duration))}`\n"
                f"{'**Prompt:** Provided' if initial_prompt else ''}"
            ).strip()
            
            await message.reply_document(str(output_path), caption=caption, parse_mode=enums.ParseMode.MARKDOWN)
            await status_msg.delete()
            
            processing_time = time.time() - start_time
            self.db.log_transcription(message.from_user.id, file_hash, self.model_manager.current_model_name, info.language, processing_time, file_size, info.duration)
            
        finally:
            # Comprehensive cleanup
            temp_path.unlink(missing_ok=True)
            if 'wav_path' in locals() and Path(wav_path).exists(): Path(wav_path).unlink()
            if 'output_path' in locals() and output_path.exists(): output_path.unlink()

    async def _convert_to_wav(self, media_path: str) -> str:
        """Converts any media file to a standardized 16kHz mono WAV file using ffmpeg."""
        output_path = f"{media_path}.wav"
        # Robust ffmpeg command for compatibility
        cmd = ['ffmpeg', '-y', '-i', media_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path]
        process = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0: raise IOError(f"FFmpeg error: {stderr.decode()}")
        return output_path

    async def _transcribe_audio_realtime(self, audio_path: str, language: Optional[str], initial_prompt: Optional[str]) -> AsyncGenerator[Tuple[Any, Any], None]:
        """Transcribes with accuracy-focused settings and yields segments in real-time."""
        # Hallmark settings with STRICTER anti-hallucination filters
        segments, info = self.model_manager.current_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language=language,
            initial_prompt=initial_prompt,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # --- STRONGER SETTINGS TO PREVENT REPETITION ---
            compression_ratio_threshold=2.0,  # Lowered from 2.4 to be more strict
            log_prob_threshold=-0.9,          # Raised from -1.0 to be more strict
            no_speech_threshold=0.6
        )
        for segment in segments:
            yield segment, info


    def _format_output(self, text: str, info, format_type: str) -> str:
        """Formats the transcription segments into plain text or SRT."""
        if format_type == 'srt':
            def to_srt_time(seconds: float) -> str:
                h, r = divmod(seconds, 3600)
                m, s = divmod(r, 60)
                return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"

            srt_lines = []
            # This is a simplified SRT generation from segments if they were available.
            # For this implementation, we use the full text with total duration.
            # A more advanced version would iterate through segments here.
            srt_lines.append(f"1\n{to_srt_time(0)} --> {to_srt_time(info.duration)}\n{text}")
            return "\n\n".join(srt_lines)
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
    print("🔥 Enhanced Telegram Whisper Bot - Colab Edition v6")
    bot_instance = None
    try:
        config = await initialize_config()
        bot_instance = ColabWhisperBot(config)
        await bot_instance.run()
        await asyncio.Event().wait()  # Keep the bot running indefinitely
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
    # Run the main asynchronous event loop
    asyncio.run(main())

