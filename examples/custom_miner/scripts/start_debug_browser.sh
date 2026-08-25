#!/usr/bin/env bash
# Start a real Chrome/Chromium in debug mode for the miner to ATTACH to. Linux only.
#
#   ./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/claude-1
#
# WHY this exists. The miner can launch Firefox itself, but a Playwright-launched
# Firefox is a recognisable automation build, and providers that fingerprint hard
# -- Google sign-in above all -- refuse it ("Couldn't sign you in"). A browser
# YOU start is not in automation mode: navigator.webdriver is false and it looks
# like the ordinary browser it is, so the sign-in works. The miner then attaches
# over CDP (Chromium only -- Firefox cannot be attached to).
#
# You start it once, log in by hand, and leave it running. The miner attaches and
# detaches freely; disconnecting never closes it, so restarting the miner keeps
# your login. Set CLAUDE_CDP=<port> (or CHATGPT_CDP=<port>) in .env to arm attach
# mode.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
  echo "Linux only (see preflight.py). On Windows use WSL2." >&2
  exit 1
fi

PORT=9222
PROFILE=""
URL="about:blank"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)    PORT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --url)     URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
PROFILE="${PROFILE:-$HOME/.hone-miner/chrome/$PORT}"

find_chrome() {
  if [ -n "${CHROME_BIN:-}" ]; then printf '%s' "$CHROME_BIN"; return; fi
  for c in google-chrome-stable google-chrome chromium chromium-browser; do
    command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return; }
  done
  # Playwright's own Chromium, if `python -m playwright install chromium` fetched one.
  for c in "${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"/chromium-*/chrome-linux/chrome; do
    [ -x "$c" ] && { printf '%s' "$c"; return; }
  done
  return 1
}

if ! CHROME="$(find_chrome)"; then
  cat >&2 <<'MSG'
No Chrome or Chromium found. Install one:
  Debian/Ubuntu:  sudo apt-get install -y chromium            # or google-chrome-stable
  Fedora/Rocky:   sudo dnf install -y chromium
  or set CHROME_BIN=/path/to/chrome
On a headless server you also need Xvfb:
  Debian/Ubuntu:  sudo apt-get install -y xvfb
MSG
  exit 1
fi

# Already running on this port? Attaching a second launcher would just open a
# tab in it and exit, which reads as "nothing happened".
if command -v curl >/dev/null 2>&1 &&
   curl -sf --max-time 2 "http://127.0.0.1:$PORT/json/version" >/dev/null; then
  echo "A debug browser is already listening on 127.0.0.1:$PORT — leaving it up."
  exit 0
fi

mkdir -p "$PROFILE"
FLAGS=(
  # Chrome 136+ refuses remote debugging on the DEFAULT profile, so an explicit
  # --user-data-dir is required, not merely tidy.
  "--user-data-dir=$PROFILE"
  "--remote-debugging-port=$PORT"
  # NEVER add --remote-debugging-address: the default binds to loopback, and the
  # port is full control of a browser holding your logged-in sessions, on a box
  # already exposing a public axon port. Reach it over an SSH tunnel instead.
  --no-first-run --no-default-browser-check
  --disable-backgrounding-occluded-windows
  --disable-renderer-backgrounding
  --disable-background-timer-throttling
  --window-size=1280,900
)
if [ "$(id -u)" -eq 0 ]; then
  echo "WARNING: running as root, so Chrome needs --no-sandbox. Prefer a normal user." >&2
  FLAGS+=(--no-sandbox)
fi

launch() {
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    "$CHROME" "${FLAGS[@]}" "$URL" &
  elif command -v xvfb-run >/dev/null 2>&1; then
    echo "No display; starting under Xvfb (reach it via VNC — see the README)."
    xvfb-run -a --server-args="-screen 0 1280x900x24" "$CHROME" "${FLAGS[@]}" "$URL" &
  else
    echo "No display and no xvfb-run. Install xvfb, or run this on a desktop." >&2
    exit 1
  fi
}

echo "chrome:  $CHROME"
echo "profile: $PROFILE"
echo "cdp:     http://127.0.0.1:$PORT"
launch
CHROME_PID=$!

for _ in $(seq 1 40); do
  if command -v curl >/dev/null 2>&1 &&
     curl -sf --max-time 2 "http://127.0.0.1:$PORT/json/version" >/dev/null; then
    echo "CDP is up on http://127.0.0.1:$PORT"
    cat <<MSG

Next:
  1. Log in to the provider in this browser (email/password or an email code —
     NOT "Continue with Google" if it rejects you). On a headless host, reach
     the window over VNC:  see "Logging in with no screen" in the README.
  2. Point .env at it:     CLAUDE_CDP=$PORT   (or CHATGPT_CDP=$PORT)
  3. Verify:               python -m solvers.doctor claude --probe
  4. Run the miner:        python examples/custom_miner/run_miner.py

Leave this browser running. The miner attaches over CDP; disconnecting never
closes it, so a miner restart keeps your login.
MSG
    wait "$CHROME_PID"
    exit $?
  fi
  sleep 0.5
done

echo "Chrome did not open a CDP port on $PORT within 20s." >&2
kill "$CHROME_PID" 2>/dev/null || true
exit 1
