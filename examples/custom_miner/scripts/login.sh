#!/usr/bin/env bash
# Log in to a provider on a machine with no screen. Linux only.
#
#   ./scripts/login.sh claude
#   ./scripts/login.sh chatgpt --profile ~/.hone-miner/firefox/chatgpt-2
#
# Playwright launches Firefox itself, so there is no CDP port to tunnel into any
# more. Instead this starts a virtual screen (Xvfb), shares it over VNC bound to
# 127.0.0.1 only, and opens the login window on it. You tunnel the VNC port in
# and type your password into a real browser.
#
# On a machine that HAS a display, skip this and run the helper directly:
#   python -m solvers.login claude
set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
  echo "Linux only (see preflight.py). On Windows use WSL2." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VNC_PORT="${VNC_PORT:-5900}"
DISPLAY_NUM="${DISPLAY_NUM:-:97}"

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 0
fi

if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  echo "You have a display; opening the window directly."
  cd "$HERE" && exec python -m solvers.login "$@"
fi

for tool in Xvfb x11vnc; do
  command -v "$tool" >/dev/null 2>&1 || {
    cat >&2 <<MSG
$tool is missing. Install both:
  Debian/Ubuntu:  sudo apt-get install -y xvfb x11vnc
  Fedora/Rocky:   sudo dnf install -y xorg-x11-server-Xvfb x11vnc

Or skip all of this: log in on a desktop and copy the profile over --
  rsync -a ~/.hone-miner/firefox/claude-1/ user@this-host:~/.hone-miner/firefox/claude-1/
MSG
    exit 1
  }
done

cleanup() {
  [ -n "${VNC_PID:-}" ] && kill "$VNC_PID" 2>/dev/null || true
  [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

Xvfb "$DISPLAY_NUM" -screen 0 1280x900x24 >/dev/null 2>&1 &
XVFB_PID=$!
sleep 1
# -localhost so the VNC port never leaves loopback: it is an unauthenticated
# view of a browser you are about to type a password into.
DISPLAY="$DISPLAY_NUM" x11vnc -display "$DISPLAY_NUM" -rfbport "$VNC_PORT" \
        -localhost -nopw -quiet -forever >/dev/null 2>&1 &
VNC_PID=$!
sleep 1

cat <<MSG

From your own machine:

    ssh -N -L ${VNC_PORT}:127.0.0.1:${VNC_PORT} $(whoami)@$(hostname -f 2>/dev/null || hostname)

then point a VNC client at 127.0.0.1:${VNC_PORT} and sign in there.
Come back to this terminal and press Enter when you are done.

MSG

cd "$HERE"
DISPLAY="$DISPLAY_NUM" python -m solvers.login "$@"
