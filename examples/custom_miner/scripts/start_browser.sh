#!/usr/bin/env bash
# Start one logged-in Chrome for a browser backend. Linux only.
#
#   ./scripts/start_browser.sh --port 9222 --profile ~/.chrome-claude-1 --url https://claude.ai
#
# One browser per account: accounts are the rate-limit unit, so a second account
# in a second profile on a second port is what doubles throughput. Log in once,
# by hand, in the window this opens; the profile directory keeps the session, so
# later runs come up already logged in.
#
# On a headless server there is no X display, so Chrome is started under Xvfb.
# Headless Chrome is deliberately NOT used: these sites treat it as a bot far
# more readily, and a CAPTCHA on a miner is a silent run of zeros.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
  echo "This script is Linux only (see preflight.py). On Windows use WSL2." >&2
  exit 1
fi

PORT=9222
PROFILE=""
URL="https://claude.ai/new"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)    PORT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --url)     URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
PROFILE="${PROFILE:-$HOME/.chrome-cdp-$PORT}"

find_chrome() {
  if [ -n "${CHROME_BIN:-}" ]; then printf '%s' "$CHROME_BIN"; return; fi
  for candidate in google-chrome-stable google-chrome chromium chromium-browser; do
    if command -v "$candidate" >/dev/null 2>&1; then command -v "$candidate"; return; fi
  done
  # Playwright's own Chromium, if the miner's `pip install playwright` pulled one.
  for candidate in "${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"/chromium-*/chrome-linux/chrome; do
    [ -x "$candidate" ] && { printf '%s' "$candidate"; return; }
  done
  return 1
}

if ! CHROME="$(find_chrome)"; then
  cat >&2 <<'MSG'
No Chrome or Chromium found. Install one:

  Debian/Ubuntu:  sudo apt-get install -y chromium            # or google-chrome-stable
  Fedora/Rocky:   sudo dnf install -y chromium
  or set CHROME_BIN=/path/to/chrome

Xvfb is needed too, for a host with no display:

  Debian/Ubuntu:  sudo apt-get install -y xvfb
  Fedora/Rocky:   sudo dnf install -y xorg-x11-server-Xvfb
MSG
  exit 1
fi

# An instance already holding this profile would just open a tab in the running
# browser and exit, leaving you to wonder why nothing happened.
if command -v curl >/dev/null 2>&1 &&
   curl -sf --max-time 2 "http://127.0.0.1:$PORT/json/version" >/dev/null; then
  echo "A browser is already listening on 127.0.0.1:$PORT — nothing to do."
  exit 0
fi

mkdir -p "$PROFILE"

FLAGS=(
  # Chrome 136+ refuses remote debugging on the DEFAULT profile, so an explicit
  # --user-data-dir is required, not just convenient.
  "--user-data-dir=$PROFILE"
  "--remote-debugging-port=$PORT"
  # Never add --remote-debugging-address: the default binds to loopback, and
  # exposing this port hands anyone full control of the browser AND its logged-in
  # sessions. This box also serves a public axon port, so keep it local and
  # reach it over an SSH tunnel if you need to.
  --no-first-run
  --no-default-browser-check
  --disable-background-timer-throttling
  --disable-renderer-backgrounding
  --disable-backgrounding-occluded-windows
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
    echo "No display; starting under Xvfb."
    xvfb-run -a --server-args="-screen 0 1280x900x24" "$CHROME" "${FLAGS[@]}" "$URL" &
  else
    echo "No display and no xvfb-run. Install xvfb (see above), or run this on a desktop." >&2
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
    echo "CDP is up."
    cat <<MSG

Next:
  1. Log in to $URL in the window that just opened. With Xvfb there is no
     visible window — reach it with:  ssh -N -L $PORT:127.0.0.1:$PORT <this-host>
     then open http://127.0.0.1:$PORT in your own browser and use the tab list.
  2. Check the selectors:  python -m solvers.doctor claude --probe
  3. Start the miner:      CLAUDE_PORTS=$PORT MINER_BACKENDS=claude \\
                             python examples/custom_miner/run_miner.py

Leave this browser running. Under systemd, give it Restart=always and the same
--user-data-dir, so a crash comes back already logged in.
MSG
    wait "$CHROME_PID"
    exit $?
  fi
  sleep 0.5
done

echo "Chrome did not open a CDP port on $PORT within 20s." >&2
kill "$CHROME_PID" 2>/dev/null || true
exit 1
