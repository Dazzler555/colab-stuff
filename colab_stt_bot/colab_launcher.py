
import asyncio
import nest_asyncio
import sys
import os

# Apply nest_asyncio for Colab compatibility
nest_asyncio.apply()

# Install requirements if needed
def install_requirements():
    required_packages = [
        "pyrogram", "tgcrypto", "faster-whisper", "yt-dlp", 
        "torch", "aiofiles", "aiohttp", "requests"
    ]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            os.system(f"pip install {package}")

def download_bot_code():
    """Download the latest bot code"""
    import urllib.request

    # You can replace this URL with your actual bot file location
    bot_url = "https://raw.githubusercontent.com/your-repo/colab_whisper_bot.py"

    try:
        urllib.request.urlretrieve(bot_url, "colab_whisper_bot.py")
        print("✅ Bot code downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Could not download bot code: {e}")
        print("📝 Please ensure colab_whisper_bot.py is in your Colab environment")
        return False

def launch_bot():
    """Launch the Colab Whisper Bot"""
    print("🚀 Enhanced Telegram Whisper Bot - Colab Launcher")
    print("=" * 60)

    # Install requirements
    install_requirements()

    # Try to download bot code if not present
    if not os.path.exists("colab_whisper_bot.py"):
        download_bot_code()

    # Import and run the bot
    if os.path.exists("colab_whisper_bot.py"):
        print("🤖 Starting bot...")
        exec(open('colab_whisper_bot.py').read())
    else:
        print("❌ Bot code not found. Please upload colab_whisper_bot.py to your Colab environment.")

if __name__ == "__main__":
    launch_bot()
