#!/bin/bash
# pycharm.sh - Auto-install and launch PyCharm with bundled Java (JBR)
# Also installs Java system-wide if missing
# Works on Linux x64

PYCHARM_VERSION="2025.1"  # Change to latest if needed
PYCHARM_EDITION="professional" # or "community"
DOWNLOAD_URL="https://download.jetbrains.com/python/pycharm-${PYCHARM_EDITION}-${PYCHARM_VERSION}.tar.gz"
INSTALL_DIR="$HOME/.pycharm"
TAR_FILE="$INSTALL_DIR/pycharm.tar.gz"

# --- STEP 1: Check & Install Java if missing ---
echo "Checking for Java..."
if type java >/dev/null 2>&1; then
    echo "✅ Java is already installed: $(java -version 2>&1 | head -n 1)"
else
    echo "⚠ Java not found. Installing OpenJDK 17..."
    if [ -x "$(command -v apt)" ]; then
        sudo apt update && sudo apt install -y openjdk-17-jdk
    elif [ -x "$(command -v dnf)" ]; then
        sudo dnf install -y java-17-openjdk-devel
    elif [ -x "$(command -v pacman)" ]; then
        sudo pacman -Sy --noconfirm jdk17-openjdk
    else
        echo "❌ No supported package manager found. Please install Java manually."
        exit 1
    fi
fi

JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export JAVA_HOME
echo "JAVA_HOME set to: $JAVA_HOME"

# --- STEP 2: Create Install Directory ---
mkdir -p "$INSTALL_DIR"

# --- STEP 3: Download PyCharm ---
echo "Downloading PyCharm ${PYCHARM_VERSION} (${PYCHARM_EDITION}) with bundled Java..."
curl -L "$DOWNLOAD_URL" -o "$TAR_FILE"

# --- STEP 4: Extract ---
echo "Extracting..."
tar -xzf "$TAR_FILE" -C "$INSTALL_DIR"
rm "$TAR_FILE"

# Find extracted folder
PYCHARM_FOLDER=$(find "$INSTALL_DIR" -maxdepth 1 -type d -name "pycharm-*" | head -n 1)

if [ -z "$PYCHARM_FOLDER" ]; then
    echo "❌ PyCharm extraction failed."
    exit 1
fi

# --- STEP 5: Launch ---
echo "✅ PyCharm installed at: $PYCHARM_FOLDER"
echo "Launching PyCharm with JAVA_HOME=$JAVA_HOME..."
"$PYCHARM_FOLDER/bin/pycharm.sh"
