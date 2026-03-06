#!/bin/bash
# Install dependencies for speech-to-cli
set -e

echo "🔧 Installing speech-to-cli dependencies..."

# System packages (ALSA tools + clipboard)
if command -v apt-get &>/dev/null; then
    sudo apt-get install -y alsa-utils xclip python3 python3-pip
elif command -v dnf &>/dev/null; then
    sudo dnf install -y alsa-utils xclip python3 python3-pip
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm alsa-utils xclip python python-pip
else
    echo "⚠️  Install manually: alsa-utils, xclip, python3, pip"
fi

# Python dependencies
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "✅ Installed! Next steps:"
echo ""
echo "  1. Set your Azure Speech key:"
echo "     export AZURE_SPEECH_KEY=\"your-key\""
echo ""
echo "  2. For Copilot CLI integration, add to ~/.copilot/mcp.json:"
echo "     {\"mcpServers\":{\"azure-speech\":{\"command\":\"python3\",\"args\":[\"$(realpath "$(dirname "$0")/mcp_speech.py")\"]}}}}"
echo ""
echo "  3. Or run standalone:  python3 $(dirname "$0")/speech.py"
