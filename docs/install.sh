#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "[*] Adding Luminous APT Repository..."
echo "deb [trusted=yes] https://aquamarine-hoshino170.github.io/C-LUMINOUS.org ./" > "$PREFIX/etc/apt/sources.list.d/luminous.list"

echo "[*] Updating package index..."
pkg update -y

echo "[*] Installing Luminous..."
pkg install luminous -y

echo "[✓] Luminous Scientific Language successfully installed!"
echo "[*] Run 'luminous --help' to get started."
