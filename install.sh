#!/usr/bin/env bash
set -e

echo "=================================================="
echo "      ❖ Installing C Luminous Language Engine ❖   "
echo "=================================================="

# Detect environment
if [ -n "$PREFIX" ] && [ -d "$PREFIX/bin" ]; then
    INSTALL_DIR="$PREFIX/bin"
else
    INSTALL_DIR="/usr/local/bin"
fi

# Build native binary
clang -O3 src/lum_runtime.c -lm -o luminous

# Install globally
mkdir -p "$INSTALL_DIR"
cp -f luminous "$INSTALL_DIR/luminous"
chmod +x "$INSTALL_DIR/luminous"

echo ""
echo "❖ [Success] C Luminous has been installed globally!"
echo "❖ Run 'luminous' anywhere in your terminal to start the REPL."
echo "=================================================="
