#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "[*] Configuring Luminous APT repository..."
echo "deb [trusted=yes] https://aquamarine-hoshino170.github.io/C-LUMINOUS.org ./" > "$PREFIX/etc/apt/sources.list.d/luminous.list"

echo "[*] Updating package index..."
pkg update -y

echo "[*] Installing Luminous package..."
pkg install luminous -y

echo "[✓] Installation complete! Type 'luminous --help' to use."
