#!/usr/bin/env bash
# Start a real Chrome/Chromium in debug mode for the miner to ATTACH to. Linux only.
#
#   ./scripts/start_debug_browser.sh --port 9222 --profile ~/.hone-miner/chrome/claude-1
#
# WHY this exists. A browser launched by an automation driver announces itself
# as one, and providers that fingerprint hard -- Google sign-in above all --
# refuse it ("Couldn't sign you in. This browser or app may not be secure.").
# A browser YOU start is not in automation mode: navigator.webdriver is false
# and it is the ordinary browser it appears to be, so the sign-in works. The
# miner then attaches over CDP, which is a Chromium protocol -- that is what
# fixes the browser choice here.
#
# You start it once, sign in by hand, and leave it running. The miner attaches
# and detaches freely; disconnecting never closes it, so restarting the miner
# keeps your login. On the default port nothing needs configuring; on any other
# port set CLAUDE_CDP=<port> (or CHATGPT_CDP=<port>) in .env.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
  echo "Linux only (see preflight.py). On Windows use WSL2." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=9222
PROFILE=""
URL="about:blank"
DISPLAY_NUM="${DISPLAY_NUM:-}"   # X display for the headless path; see below

need_value() {  # `set -u` would otherwise abort with "$2: unbound variable"
  [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
}
while [ $# -gt 0 ]; do
  case "$1" in
    --port)    need_value "$@"; PORT="$2"; shift 2 ;;
    --profile) need_value "$@"; PROFILE="$2"; shift 2 ;;
    --url)     need_value "$@"; URL="$2"; shift 2 ;;
    --display) need_value "$@"; DISPLAY_NUM="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
PROFILE="${PROFILE:-$HOME/.hone-miner/chrome/$PORT}"
# A FIXED display per port, not xvfb-run's `-a` search. `-a` picks whatever is
# free and never tells you which, so the VNC command you need afterwards is
# unknowable -- and a second browser on another port silently lands elsewhere.
# Deriving it from the port keeps two browsers apart and keeps both printable.
if [ -z "$DISPLAY_NUM" ]; then
  DISPLAY_NUM=$((PORT - 9123))
  [ "$DISPLAY_NUM" -ge 1 ] 2>/dev/null || DISPLAY_NUM=99
fi
# xvfb-run's default auth file is private to the command it runs, so a separate
# x11vnc has no cookie and dies with "No protocol specified". Put it somewhere
# both can read, and print the path.
XAUTH="$PROFILE/.Xauthority"

# Both the double-launch guard and the readiness probe below ask the CDP port
# whether it is up. Without curl the guard fails open and the probe can never
# succeed, so the script would kill a perfectly healthy browser after 20s.
command -v curl >/dev/null 2>&1 || {
  echo "curl is required (used to check the CDP port). sudo apt-get install -y curl" >&2
  exit 1
}

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
if curl -sf --max-time 2 "http://127.0.0.1:$PORT/json/version" >/dev/null; then
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
  # Quiet the alarming-but-harmless server noise: no GPU on a headless/RDP box
  # (GpuControl errors), and log only fatal messages so the GCM push-registration
  # and TensorFlow-delegate lines stop looking like failures. CDP is unaffected.
  --disable-gpu
  --log-level=3
  --disable-features=OptimizationHints,MediaRouter,Translate
)
if [ "$(id -u)" -eq 0 ]; then
  echo "WARNING: running as root, so Chrome needs --no-sandbox. Prefer a normal user." >&2
  FLAGS+=(--no-sandbox)
fi

launch() {
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    "$CHROME" "${FLAGS[@]}" "$URL" &
  elif command -v xvfb-run >/dev/null 2>&1; then
    echo "No display; starting under Xvfb on :$DISPLAY_NUM (reach it via VNC)."
    xvfb-run -n "$DISPLAY_NUM" -f "$XAUTH" \
             --server-args="-screen 0 1280x900x24" "$CHROME" "${FLAGS[@]}" "$URL" &
  else
    echo "No display and no xvfb-run. Install xvfb, or run this on a desktop." >&2
    exit 1
  fi
}

echo "chrome:  $CHROME"
echo "profile: $PROFILE"
echo "cdp:     http://127.0.0.1:$PORT"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo "display: :$DISPLAY_NUM (xauth $XAUTH)"
fi
launch
LAUNCHER_PID=$!   # xvfb-run's pid on a headless host, not Chrome's

for _ in $(seq 1 40); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/json/version" >/dev/null; then
    echo "CDP is up on http://127.0.0.1:$PORT"
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
      cat <<MSG

There is no screen here, so reach the browser window over VNC. In a SECOND
terminal on this host:

  x11vnc -display :$DISPLAY_NUM -auth $XAUTH -rfbport 5900 -localhost -nopw -forever

then from YOUR machine:

  ssh -N -L 5900:127.0.0.1:5900 $(whoami)@$(hostname -f 2>/dev/null || hostname)

and point a VNC client at 127.0.0.1:5900. The -localhost and the tunnel matter:
that screen is an unauthenticated view of a browser you are about to type a
password into.
MSG
    fi
    cat <<MSG

Next:
  1. Sign in to the provider in this browser, by hand. Prefer email plus a
     one-time code — "Continue with Google" is the sign-in most likely to be
     refused.
  2. If this is not port 9222, point .env at it: CLAUDE_CDP=$PORT
  3. Verify:   cd $HERE && python -m solvers.doctor claude --probe
  4. Run:      python run_miner.py        # from $HERE

Leave this browser running; this terminal is now busy, so use another one.
The miner attaches over CDP and never closes it, so a restart keeps your login.
MSG
    wait "$LAUNCHER_PID"
    exit $?
  fi
  sleep 0.5
done

echo "Chrome did not open a CDP port on $PORT within 20s." >&2
# Under Xvfb this kills the wrapper, which usually takes Chrome with it -- but
# not always, so say how to check rather than pretending it is gone.
kill "$LAUNCHER_PID" 2>/dev/null || true
echo "If port $PORT is still busy, a browser was orphaned: check with" >&2
echo "  curl -s http://127.0.0.1:$PORT/json/version" >&2
exit 1
