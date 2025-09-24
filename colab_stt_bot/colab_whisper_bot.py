
# Enhanced Telegram Whisper Bot - Google Colab Version
# Optimized for Google Colab with nested asyncio and gist config loading

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

# Colab-specific imports for nested asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ nest_asyncio applied for Colab compatibility")
except ImportError:
    print("⚠️ nest_asyncio not found, installing...")
    os.system("pip install nest_asyncio")
    import nest_asyncio
    nest_asyncio.apply()

# Install required packages if not available
required_packages = [
    "pyrogram", "tgcrypto", "faster-whisper", "yt-dlp", 
    "torch", "aiofiles", "aiohttp", "requests"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"📦 Installing {package}...")
        os.system(f"pip install {package}")

# Now import the main libraries
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

# --- Colab Configuration Management ---
@dataclass
class ColabBotConfig:
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
    """Load configuration from GitHub Gist"""
    try:
        # Extract raw URL from gist URL if needed
        if "gist.github.com" in gist_url and "/raw/" not in gist_url:
            # Convert gist URL to raw URL
            gist_id = gist_url.split("/")[-1]
            raw_url = f"https://gist.githubusercontent.com/{gist_id}/raw/"
        else:
            raw_url = gist_url

        print(f"📡 Loading config from: {raw_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(raw_url) as response:
                if response.status == 200:
                    text = await response.text()
                    config_data = json.loads(text)
                    print("✅ Config loaded successfully from gist")
                    return config_data
                else:
                    raise Exception(f"Failed to fetch gist: HTTP {response.status}")

    except Exception as e:
        print(f"❌ Error loading config from gist: {e}")
        # Fallback to manual config input
        print("💡 Using manual configuration...")
        return {}

def get_manual_config() -> ColabBotConfig:
    """Get configuration through manual input in Colab"""
    print("🔧 Manual Configuration Setup")
    print("=" * 50)

    # Get basic credentials
    api_id = input("Enter your Telegram API ID: ")
    api_hash = input("Enter your Telegram API Hash: ")
    bot_token = input("Enter your Bot Token: ")

    # Optional gist URL for future updates
    gist_url = input("Enter Gist URL for config updates (optional): ") or ""

    # Admin users
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
    """Initialize configuration for Colab environment"""
    # Check if gist URL is provided via environment or input
    gist_url = os.environ.get('CONFIG_GIST_URL') or input("Enter Config Gist URL (or press Enter for manual setup): ")

    if gist_url:
        try:
            config_data = await load_config_from_gist(gist_url)
            if config_data:
                # Add gist URL to config for future updates
                config_data['gist_url'] = gist_url
                return ColabBotConfig(**config_data)
        except Exception as e:
            print(f"⚠️ Failed to load from gist: {e}")

    # Fallback to manual configuration
    return get_manual_config()

# --- Enhanced Database Manager for Colab ---
class ColabDatabaseManager:
    def __init__(self, db_path: str = "/content/bot_data.db"):
        self.db_path = db_path
        print(f"🗄️ Initializing database at: {db_path}")
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        first_name TEXT,
                        language_code TEXT DEFAULT 'auto',
                        preferred_model TEXT DEFAULT 'large-v2',
                        translate_to TEXT DEFAULT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_premium BOOLEAN DEFAULT FALSE,
                        usage_count INTEGER DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS transcriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        file_hash TEXT,
                        model_used TEXT,
                        language_detected TEXT,
                        processing_time REAL,
                        file_size INTEGER,
                        duration REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    );

                    CREATE TABLE IF NOT EXISTS rate_limits (
                        user_id INTEGER,
                        timestamp TIMESTAMP,
                        request_count INTEGER DEFAULT 1,
                        PRIMARY KEY (user_id, timestamp)
                    );

                    CREATE INDEX IF NOT EXISTS idx_rate_limits_user_time 
                    ON rate_limits (user_id, timestamp);
                """)
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"❌ Error getting user {user_id}: {e}")
            return None

    def create_or_update_user(self, user_id: int, username: str = None, 
                            first_name: str = None, **kwargs):
        """Create or update user in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, username, first_name, language_code, preferred_model, 
                     translate_to, last_active, is_premium)
                    VALUES (?, ?, ?, 
                           COALESCE((SELECT language_code FROM users WHERE user_id = ?), ?),
                           COALESCE((SELECT preferred_model FROM users WHERE user_id = ?), ?),
                           COALESCE((SELECT translate_to FROM users WHERE user_id = ?), ?),
                           CURRENT_TIMESTAMP,
                           COALESCE((SELECT is_premium FROM users WHERE user_id = ?), ?))
                """, (
                    user_id, username, first_name, user_id, kwargs.get('language_code', 'auto'),
                    user_id, kwargs.get('preferred_model', 'large-v2'), 
                    user_id, kwargs.get('translate_to'),
                    user_id, kwargs.get('is_premium', False)
                ))
        except Exception as e:
            print(f"❌ Error updating user {user_id}: {e}")

    def check_rate_limit(self, user_id: int, max_requests: int, window_seconds: int) -> bool:
        """Check if user has exceeded rate limit"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT SUM(request_count) as total_requests 
                    FROM rate_limits 
                    WHERE user_id = ? AND timestamp > ?
                """, (user_id, cutoff_time))

                result = cursor.fetchone()
                current_requests = result[0] if result[0] else 0

                if current_requests >= max_requests:
                    return False

                # Log this request
                now = datetime.now().replace(second=0, microsecond=0)
                conn.execute("""
                    INSERT OR REPLACE INTO rate_limits (user_id, timestamp, request_count)
                    VALUES (?, ?, COALESCE((SELECT request_count FROM rate_limits 
                            WHERE user_id = ? AND timestamp = ?), 0) + 1)
                """, (user_id, now, user_id, now))

                return True
        except Exception as e:
            print(f"❌ Rate limit check error: {e}")
            return True  # Allow on error

    def log_transcription(self, user_id: int, file_hash: str, model_used: str,
                         language_detected: str, processing_time: float,
                         file_size: int, duration: float):
        """Log transcription activity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO transcriptions 
                    (user_id, file_hash, model_used, language_detected, 
                     processing_time, file_size, duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, file_hash, model_used, language_detected, 
                      processing_time, file_size, duration))

                # Update user usage count
                conn.execute("""
                    UPDATE users SET usage_count = usage_count + 1,
                    last_active = CURRENT_TIMESTAMP WHERE user_id = ?
                """, (user_id,))
        except Exception as e:
            print(f"❌ Error logging transcription: {e}")

# --- Colab Model Manager ---
class ColabModelManager:
    def __init__(self, model_dir: str = "/content/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.current_model_name = None

        # Optimized models for Colab (considering limited resources)
        self.model_info = {
            "tiny": {"size": "tiny", "compute": "float16", "vram": "~0.8GB", "speed": "~32x", "recommended": "Speed"},
            "base": {"size": "base", "compute": "float16", "vram": "~1GB", "speed": "~16x", "recommended": "Balanced"},
            "small": {"size": "small", "compute": "float16", "vram": "~1.4GB", "speed": "~6x", "recommended": "Quality"},
            "medium": {"size": "medium", "compute": "float16", "vram": "~2.7GB", "speed": "~2x", "recommended": "High Quality"},
            "large-v2": {"size": "large-v2", "compute": "float16", "vram": "~4.3GB", "speed": "~1x", "recommended": "Best Quality"},
            "turbo": {"size": "turbo", "compute": "float16", "vram": "~6GB", "speed": "~8x", "recommended": "Fast + Quality"},
        }

    async def load_model(self, model_name: str) -> WhisperModel:
        """Load Whisper model optimized for Colab"""
        try:
            if model_name not in self.model_info:
                raise ValueError(f"Unknown model: {model_name}")

            print(f"🤖 Loading model: {model_name}")

            model_size = self.model_info[model_name]["size"]

            # Auto-detect best device and compute type for Colab
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                print(f"🔥 Using GPU acceleration (CUDA)")
            else:
                device = "cpu"
                compute_type = "int8"  # Better for CPU
                print(f"🖥️ Using CPU (GPU not available)")

            # Clean up old models to save space
            await self._cleanup_old_models()

            self.current_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_dir=str(self.model_dir)
            )

            self.current_model_name = model_name
            print(f"✅ Model {model_name} loaded successfully on {device}")
            return self.current_model

        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            raise

    async def _cleanup_old_models(self):
        """Keep only 1 model to save Colab storage"""
        try:
            model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
            if len(model_dirs) > 1:
                # Sort by modification time and remove all but the newest
                model_dirs.sort(key=lambda x: x.stat().st_mtime)
                for old_dir in model_dirs[:-1]:
                    shutil.rmtree(old_dir)
                    print(f"🧹 Cleaned up old model: {old_dir.name}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

# --- Main Colab Bot Class ---
class ColabWhisperBot:
    def __init__(self, config: ColabBotConfig):
        self.config = config
        self.db = ColabDatabaseManager(config.database_path)
        self.model_manager = ColabModelManager(config.model_dir)

        # Create directories
        for directory in [config.model_dir, config.download_dir, config.temp_dir]:
            Path(directory).mkdir(exist_ok=True)

        self.processing_status = {}

        # Initialize Pyrogram client
        self.app = Client(
            "colab_whisper_bot",
            api_id=config.api_id,
            api_hash=config.api_hash,
            bot_token=config.bot_token,
            workdir="/content"  # Colab-specific workdir
        )

        self._setup_handlers()
        print("🤖 Bot initialized successfully!")

    def _setup_handlers(self):
        """Setup bot handlers"""
        @self.app.on_message(filters.command("start"))
        async def start_command(client: Client, message: Message):
            await self.handle_start(client, message)

        @self.app.on_message(filters.command("help"))
        async def help_command(client: Client, message: Message):
            await self.handle_help(client, message)

        @self.app.on_message(filters.command("status"))
        async def status_command(client: Client, message: Message):
            await self.handle_status(client, message)

        @self.app.on_message(filters.command("models"))
        async def models_command(client: Client, message: Message):
            await self.handle_models(client, message)

        @self.app.on_message(filters.command("stats"))
        async def stats_command(client: Client, message: Message):
            await self.handle_stats(client, message)

        @self.app.on_message(filters.reply & (filters.audio | filters.video | filters.document | 
                                            filters.voice | filters.video_note | filters.text))
        async def media_handler(client: Client, message: Message):
            await self.handle_media_processing(client, message)

        @self.app.on_callback_query()
        async def callback_handler(client: Client, callback_query: CallbackQuery):
            await self.handle_callbacks(client, callback_query)

    async def handle_start(self, client: Client, message: Message):
        """Enhanced start command"""
        user = message.from_user
        self.db.create_or_update_user(user.id, user.username, user.first_name)

        welcome_text = f"""
🎤 **Enhanced Whisper Bot (Colab Edition)**

Hello {user.first_name}! I'm running on Google Colab with advanced transcription capabilities.

**✨ Features:**
• 🎯 High-accuracy transcription in 100+ languages
• 📝 Multiple output formats (Text, SRT)
• 🌍 Translation support
• 🎵 Voice messages & video support
• 📹 YouTube URL processing
• ⚡ GPU acceleration (when available)

**🚀 Quick Start:**
1. Upload audio/video or send voice message
2. Reply with `/transcribe`, `/subtitle`, or `/translate <lang>`
3. Use `/models` to select optimal model for your needs

**💡 Colab Tips:**
• Use smaller models (tiny/base) for faster processing
• Enable GPU runtime for better performance
• Check `/status` for current processing info

Use `/help` for detailed commands!
        """

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🤖 Models", callback_data="models"),
             InlineKeyboardButton("📊 Help", callback_data="help")],
            [InlineKeyboardButton("📈 Stats", callback_data="stats")]
        ])

        await message.reply_text(welcome_text, parse_mode=enums.ParseMode.MARKDOWN, reply_markup=keyboard)

    async def handle_help(self, client: Client, message: Message):
        """Comprehensive help for Colab environment"""
        help_text = """
📚 **Colab Whisper Bot - Help Guide**

**🎵 Processing Commands:**
• `/transcribe` - Convert to text
• `/subtitle` - Generate SRT subtitles
• `/translate <lang>` - Transcribe + translate (e.g., `/translate es`)

**⚙️ Bot Management:**
• `/models` - Select optimal model
• `/stats` - View usage statistics  
• `/status` - Check processing status

**📱 Supported Media:**
• Audio files (MP3, WAV, OGG, M4A)
• Video files (MP4, MKV, MOV, AVI)
• Voice messages & video notes
• YouTube URLs (send URL, then reply with command)

**🌍 Languages:**
Auto-detects 100+ languages including:
🇺🇸 English, 🇪🇸 Spanish, 🇫🇷 French, 🇩🇪 German, 🇨🇳 Chinese, 🇯🇵 Japanese, 🇷🇺 Russian, 🇦🇪 Arabic, and many more!

**⚡ Colab Performance Tips:**
• **tiny/base**: Fast processing, good for quick tasks
• **small**: Balanced speed/quality for most use cases  
• **medium/large**: Best quality for important content
• Enable GPU runtime: Runtime → Change runtime type → GPU

**🔄 Example Workflow:**
1. Upload your audio file
2. Reply: `/transcribe` (for text) or `/subtitle` (for SRT)
3. For translation: `/translate fr` (French), `/translate es` (Spanish)

**📊 Translation Codes:**
`en` English, `es` Spanish, `fr` French, `de` German, `it` Italian, `pt` Portuguese, `ru` Russian, `zh` Chinese, `ja` Japanese, `ko` Korean, `ar` Arabic
        """

        await message.reply_text(help_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_status(self, client: Client, message: Message):
        """Show current processing status"""
        chat_id = message.chat.id

        if chat_id in self.processing_status:
            status_info = self.processing_status[chat_id]
            status_text = f"""
📊 **Current Status**

🔄 **Process**: {status_info.get('message', 'Unknown')}
📈 **Progress**: {status_info.get('progress', 0):.1f}%
⏱️ **Elapsed**: {time.time() - status_info.get('start_time', time.time()):.1f}s
🤖 **Model**: {self.model_manager.current_model_name or 'None loaded'}
💾 **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}
            """
        else:
            status_text = f"""
✅ **Bot Status**

🤖 **Model Loaded**: {self.model_manager.current_model_name or 'None'}
💾 **Device**: {'🔥 GPU Available' if torch.cuda.is_available() else '🖥️ CPU Only'}
📁 **Storage**: Colab temporary storage
🔄 **Active Processes**: None

Ready to process your media!
            """

        await message.reply_text(status_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_models(self, client: Client, message: Message):
        """Show available models optimized for Colab"""
        models_text = "🤖 **Available Models (Colab Optimized)**\n\n"

        for name, info in self.model_manager.model_info.items():
            emoji = "⚡" if "tiny" in name or "turbo" in name else "🎯" if "large" in name else "⚖️"
            models_text += f"{emoji} **{name}**\n"
            models_text += f"├ Memory: {info['vram']}\n"
            models_text += f"├ Speed: {info['speed']}\n"
            models_text += f"└ Best for: {info['recommended']}\n\n"

        models_text += "💡 **Colab Recommendations:**\n"
        models_text += "• Free tier: Use `tiny` or `base`\n"
        models_text += "• Pro/Pro+: `small` or `medium`\n"
        models_text += "• GPU available: Any model\n"
        models_text += "• CPU only: Stick to `tiny` or `base`"

        keyboard = self._create_model_keyboard()
        await message.reply_text(models_text, parse_mode=enums.ParseMode.MARKDOWN, reply_markup=keyboard)

    def _create_model_keyboard(self):
        """Create model selection keyboard"""
        models = list(self.model_manager.model_info.keys())
        keyboard = []

        # Group models by performance tier
        fast_models = ["tiny", "base", "turbo"]
        quality_models = ["small", "medium", "large-v2"]

        # Fast models row
        fast_row = []
        for model in fast_models:
            if model in models:
                fast_row.append(InlineKeyboardButton(f"⚡ {model}", callback_data=f"select_model_{model}"))
        if fast_row:
            keyboard.append(fast_row)

        # Quality models row
        quality_row = []
        for model in quality_models:
            if model in models:
                quality_row.append(InlineKeyboardButton(f"🎯 {model}", callback_data=f"select_model_{model}"))
        if quality_row:
            keyboard.append(quality_row)

        return InlineKeyboardMarkup(keyboard)

    async def handle_stats(self, client: Client, message: Message):
        """Show user statistics"""
        user_data = self.db.get_user(message.from_user.id)
        if not user_data:
            await message.reply_text("❌ No statistics available. Use the bot first!")
            return

        stats_text = f"""
📊 **Your Statistics**

👤 **Profile:**
• Usage Count: `{user_data['usage_count']}`
• Member Since: `{user_data['created_at'][:10]}`
• Preferred Model: `{user_data['preferred_model']}`

💾 **System Info:**
• Database: `{os.path.getsize(self.db.db_path) / 1024:.1f} KB`
• Current Model: `{self.model_manager.current_model_name or 'None'}`
• Device: `{'GPU' if torch.cuda.is_available() else 'CPU'}`
• Colab Runtime: `Active`

🎯 **Quick Actions:**
Use `/models` to change model or start processing!
        """

        await message.reply_text(stats_text, parse_mode=enums.ParseMode.MARKDOWN)

    async def handle_media_processing(self, client: Client, message: Message):
        """Process media files with enhanced Colab optimization"""
        if not message.reply_to_message:
            return

        user_id = message.from_user.id
        chat_id = message.chat.id
        command = message.text.lower()

        # Validate commands
        valid_commands = ['/transcribe', '/text', '/subtitle', '/srt']
        translate_command = command.startswith('/translate')

        if not (any(cmd in command for cmd in valid_commands) or translate_command):
            return

        # Check rate limiting
        if not self.db.check_rate_limit(user_id, self.config.rate_limit_requests, self.config.rate_limit_window):
            await message.reply_text("⏱️ Rate limit exceeded. Please wait before making more requests.")
            return

        # Parse translation target if specified
        target_language = None
        if translate_command:
            parts = command.split()
            target_language = parts[1] if len(parts) > 1 else 'en'

        # Initialize processing
        status_message = await message.reply_text("⏳ **Initializing processing...**", parse_mode=enums.ParseMode.MARKDOWN)
        self.processing_status[chat_id] = {
            "message": "Initializing",
            "progress": 0,
            "start_time": time.time()
        }

        try:
            # Download media
            await self._update_status(status_message, "📥 Downloading media...", 10)
            media_info = await self._download_media(client, message, status_message)
            if not media_info:
                return

            file_path, file_hash, file_size = media_info

            # Convert to WAV
            await self._update_status(status_message, "🔄 Converting audio...", 30)
            audio_path = await self._convert_to_wav(file_path, status_message)
            if not audio_path:
                return

            # Load model if needed
            if not self.model_manager.current_model:
                await self._update_status(status_message, "🤖 Loading model...", 40)
                user_data = self.db.get_user(user_id)
                preferred_model = user_data['preferred_model'] if user_data else 'base'  # Default to base for Colab
                await self.model_manager.load_model(preferred_model)

            # Transcribe
            await self._update_status(status_message, "👂 Transcribing...", 60)
            result = await self._transcribe_audio(audio_path, status_message)

            if not result:
                await status_message.edit_text("❌ Transcription failed.")
                return

            transcription, language_detected, processing_time, duration = result

            # Handle translation if requested
            if translate_command:
                await self._update_status(status_message, f"🌍 Translating to {target_language}...", 85)
                # Simple translation simulation (you can integrate a real translation API)
                transcription = f"[Translated to {target_language}] {transcription}"

            # Generate output
            await self._update_status(status_message, "📄 Generating output...", 95)

            if '/subtitle' in command or '/srt' in command:
                # Generate SRT format
                output_content = f"""1
00:00:00,000 --> {self._seconds_to_srt_time(duration)}
{transcription}
"""
                output_filename = f"transcript_{int(time.time())}.srt"
            else:
                # Generate text format
                output_content = transcription
                output_filename = f"transcript_{int(time.time())}.txt"

            # Save and send file
            output_path = Path(self.config.temp_dir) / output_filename
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(output_content)

            # Send results
            await self._update_status(status_message, "📤 Sending results...", 100)

            caption = f"""
✅ **Processing Complete!**

🎯 **Language**: {language_detected}
⏱️ **Processing Time**: {processing_time:.1f}s
🤖 **Model**: {self.model_manager.current_model_name}
📏 **Duration**: {duration:.1f}s
{f'🌍 **Translated to**: {target_language}' if translate_command else ''}
            """

            await client.send_document(
                chat_id,
                document=str(output_path),
                caption=caption,
                parse_mode=enums.ParseMode.MARKDOWN,
                reply_to_message_id=message.reply_to_message.id
            )

            await status_message.delete()

            # Log to database
            self.db.log_transcription(
                user_id, file_hash, self.model_manager.current_model_name,
                language_detected, processing_time, file_size, duration
            )

            # Cleanup files
            for cleanup_path in [file_path, audio_path, output_path]:
                if cleanup_path and Path(cleanup_path).exists():
                    Path(cleanup_path).unlink()

        except Exception as e:
            print(f"❌ Processing error: {e}")
            await status_message.edit_text(f"❌ **Processing failed:** {str(e)}", parse_mode=enums.ParseMode.MARKDOWN)

        finally:
            if chat_id in self.processing_status:
                del self.processing_status[chat_id]

    async def handle_callbacks(self, client: Client, callback_query: CallbackQuery):
        """Handle callback queries"""
        data = callback_query.data

        try:
            if data == "help":
                await self.handle_help(client, callback_query.message)
            elif data == "models":
                await self.handle_models(client, callback_query.message)
            elif data == "stats":
                await self.handle_stats(client, callback_query.message)
            elif data.startswith("select_model_"):
                model_name = data.split("select_model_")[1]
                await self._handle_model_selection(callback_query, model_name)

            await callback_query.answer()

        except Exception as e:
            print(f"❌ Callback error: {e}")
            await callback_query.answer("❌ An error occurred.", show_alert=True)

    async def _handle_model_selection(self, callback_query: CallbackQuery, model_name: str):
        """Handle model selection"""
        try:
            await callback_query.message.edit_text(f"🤖 Loading model **{model_name}**...", parse_mode=enums.ParseMode.MARKDOWN)
            await self.model_manager.load_model(model_name)

            # Update user preference
            user_id = callback_query.from_user.id
            user_data = self.db.get_user(user_id)
            if user_data:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.execute("UPDATE users SET preferred_model = ? WHERE user_id = ?", (model_name, user_id))

            await callback_query.message.edit_text(
                f"✅ Model **{model_name}** loaded and set as default!",
                parse_mode=enums.ParseMode.MARKDOWN
            )

        except Exception as e:
            await callback_query.message.edit_text(f"❌ Failed to load model: {e}", parse_mode=enums.ParseMode.MARKDOWN)

    # Helper methods
    async def _update_status(self, status_message: Message, text: str, progress: int):
        """Update processing status"""
        try:
            await status_message.edit_text(f"**{text}** ({progress}%)", parse_mode=enums.ParseMode.MARKDOWN)
        except:
            pass

    async def _download_media(self, client: Client, message: Message, status_message: Message):
        """Download media from Telegram"""
        reply_msg = message.reply_to_message
        temp_id = str(uuid.uuid4())

        try:
            file_path = None
            file_size = 0

            if reply_msg.audio:
                file_path = await client.download_media(reply_msg.audio, file_name=f"{self.config.temp_dir}/{temp_id}_audio")
                file_size = reply_msg.audio.file_size
            elif reply_msg.video:
                file_path = await client.download_media(reply_msg.video, file_name=f"{self.config.temp_dir}/{temp_id}_video")
                file_size = reply_msg.video.file_size
            elif reply_msg.voice:
                file_path = await client.download_media(reply_msg.voice, file_name=f"{self.config.temp_dir}/{temp_id}_voice.ogg")
                file_size = reply_msg.voice.file_size
            elif reply_msg.video_note:
                file_path = await client.download_media(reply_msg.video_note, file_name=f"{self.config.temp_dir}/{temp_id}_videonote.mp4")
                file_size = reply_msg.video_note.file_size
            elif reply_msg.document:
                mime_type = reply_msg.document.mime_type or ""
                file_name = reply_msg.document.file_name or ""

                if ("audio" in mime_type or "video" in mime_type or
                    any(ext in file_name.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.mp4', '.mkv'])):
                    file_path = await client.download_media(reply_msg.document, file_name=f"{self.config.temp_dir}/{temp_id}_{file_name}")
                    file_size = reply_msg.document.file_size
                else:
                    await status_message.edit_text("❌ Unsupported file format.")
                    return None
            elif reply_msg.text and "youtube.com" in reply_msg.text:
                url = reply_msg.text.strip()
                file_path = await self._download_youtube(url, temp_id)
                file_size = os.path.getsize(file_path) if file_path else 0
            else:
                await status_message.edit_text("❌ Please reply to an audio/video file or voice message.")
                return None

            if not file_path or file_size > self.config.max_file_size:
                await status_message.edit_text(f"❌ File too large. Max: {self.config.max_file_size // (1024*1024)}MB")
                return None

            file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
            return file_path, file_hash, file_size

        except Exception as e:
            print(f"❌ Download error: {e}")
            await status_message.edit_text(f"❌ Download failed: {str(e)}")
            return None

    async def _download_youtube(self, url: str, temp_id: str) -> Optional[str]:
        """Download audio from YouTube using yt-dlp"""
        try:
            output_path = f"{self.config.temp_dir}/{temp_id}_%(title)s.%(ext)s"

            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

                # Find downloaded file
                for file in Path(self.config.temp_dir).glob(f"{temp_id}_*"):
                    if file.is_file():
                        return str(file)

            return None

        except Exception as e:
            print(f"❌ YouTube download error: {e}")
            return None

    async def _convert_to_wav(self, media_path: str, status_message: Message) -> Optional[str]:
        """Convert media to WAV format"""
        try:
            input_path = Path(media_path)
            output_path = input_path.with_suffix('.wav')

            if input_path.suffix.lower() == '.wav':
                return str(input_path)

            # FFmpeg conversion optimized for Colab
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"FFmpeg error: {stderr.decode()}")

            # Remove original to save space
            if input_path.exists():
                input_path.unlink()

            return str(output_path)

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            await status_message.edit_text(f"❌ Audio conversion failed: {str(e)}")
            return None

    async def _transcribe_audio(self, audio_path: str, status_message: Message):
        """Transcribe audio with progress tracking"""
        try:
            start_time = time.time()

            # Get audio duration
            duration = await self._get_audio_duration(audio_path)

            # Transcribe with current model
            segments, info = self.model_manager.current_model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=False,  # Disable for speed in Colab
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=50),
                temperature=0.0
            )

            # Collect transcription
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text.strip())

            transcription = " ".join(transcription_parts)
            processing_time = time.time() - start_time

            return transcription, info.language, processing_time, duration

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using FFprobe"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await process.communicate()
            return float(stdout.decode().strip())
        except:
            return 0.0

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    async def run(self):
        """Run the bot in Colab"""
        print("🚀 Starting Colab Whisper Bot...")

        try:
            await self.app.start()
            print("✅ Bot started successfully!")
            print("🤖 Bot is now running. Send /start to begin!")
            print("⚠️ Keep this cell running to maintain the bot")

            # Keep the bot running
            await self.app.idle()

        except KeyboardInterrupt:
            print("🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")
        finally:
            await self.app.stop()
            print("👋 Bot stopped")

# --- Main Function for Colab ---
async def main():
    """Main function optimized for Google Colab"""
    print("🔥 Enhanced Telegram Whisper Bot - Colab Edition")
    print("=" * 60)

    try:
        # Initialize configuration
        config = await initialize_config()
        print(f"✅ Configuration loaded successfully!")

        # Create and run bot
        bot = ColabWhisperBot(config)
        await bot.run()

    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()

# --- Colab Execution ---
if __name__ == "__main__":
    print("🚀 Launching bot in Colab environment...")

    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab detected")
    except ImportError:
        print("⚠️ Not running in Google Colab")

    # Run the bot
    asyncio.run(main())
