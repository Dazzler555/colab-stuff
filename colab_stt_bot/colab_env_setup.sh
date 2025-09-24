
#!/bin/bash
# Colab Environment Setup for Whisper Bot

echo "🔧 Setting up Colab environment for Whisper Bot..."

# Update system packages
apt-get update -qq

# Install FFmpeg (usually already available in Colab)
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 Installing FFmpeg..."
    apt-get install -y ffmpeg
else
    echo "✅ FFmpeg already available"
fi

# Install Python packages
echo "📦 Installing Python dependencies..."
pip install --quiet --upgrade     nest_asyncio     pyrogram     tgcrypto     faster-whisper     yt-dlp     torch     aiofiles     aiohttp     requests

# Create necessary directories
mkdir -p /content/models /content/downloads /content/temp

echo "✅ Environment setup complete!"
echo "🚀 Ready to run the Whisper Bot!"
