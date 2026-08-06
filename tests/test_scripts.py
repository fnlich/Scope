"""Public entrypoint, shell, and documentation checks."""

from __future__ import annotations

import http.server
import importlib.util
import os
import shutil
import subprocess
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
RUN_VALIDATOR = SCRIPTS / "run_validator.py"


def _load_run_validator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_run_validator_under_test", RUN_VALIDATOR
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_validator_fails_closed_without_problem_source(monkeypatch):
    module = _load_run_validator()
    settings = types.SimpleNamespace(problem_server_url="")
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    with pytest.raises(SystemExit, match="PROBLEM_SERVER_URL"):
        module.main()


SHELL_SCRIPTS = [
    REPO_ROOT / "setup_validator.sh",
    REPO_ROOT / "start_validator.sh",
    REPO_ROOT / "start_demo_miner.sh",
    SCRIPTS / "preflight_validator.sh",
    SCRIPTS / "register_testnet.sh",
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_are_valid_bash(script: Path):
    assert script.exists()
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_start_script_launches_only_the_validator():
    text = (REPO_ROOT / "start_validator.sh").read_text()
    assert ".venv/bin/activate" in text
    assert "preflight_validator.sh" in text
    assert "scripts/run_validator.py" in text
    assert "run_miner" not in text
    assert "problem_server" not in text


def test_start_script_fails_closed_without_setup():
    text = (REPO_ROOT / "start_validator.sh").read_text()
    assert "run ./setup_validator.sh first" in text
    assert "using current Python" not in text


def test_setup_pulls_the_configured_immutable_sandbox_image():
    setup = (REPO_ROOT / "setup_validator.sh").read_text()
    preflight = (SCRIPTS / "preflight_validator.sh").read_text()
    assert 'preflight_validator.sh" --pull' in setup
    assert "docker pull" in preflight
    assert "@sha256:" in preflight
    assert "run_timed 20 docker info" in preflight
    assert "run_timed 45 docker run" in preflight


def test_preflight_probes_the_protected_lease_route_without_consuming_a_problem():
    preflight = (SCRIPTS / "preflight_validator.sh").read_text()
    assert 'f"{base}/v1/challenges/lease"' in preflight
    assert 'data=b"{}"' in preflight
    assert 'method="POST"' in preflight
    assert "status != 401" in preflight
    assert 'headers.get("Date", "")' in preflight
    assert "/v1/health" not in preflight


@pytest.mark.parametrize(
    "status, valid_date, succeeds",
    [(401, True, True), (200, True, False), (302, True, False), (401, False, False)],
)
def test_preflight_requires_an_unsigned_lease_rejection(
    status, valid_date, succeeds
):
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *_args):
            pass

        def date_time_string(self, timestamp=None):
            if valid_date:
                return super().date_time_string(timestamp)
            return "not-a-date"

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            **os.environ,
            "EXECUTOR": "local",
            "PROBLEM_SERVER_URL": f"http://127.0.0.1:{server.server_port}",
            "SUBTENSOR_NETWORK": "test",
        }
        result = subprocess.run(
            ["bash", str(SCRIPTS / "preflight_validator.sh")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (result.returncode == 0) is succeeds
    assert received == {"path": "/v1/challenges/lease", "body": b"{}"}


def test_register_script_is_runbook_only():
    text = (SCRIPTS / "register_testnet.sh").read_text()
    assert "exit 0" in text
    for keyword in ("new_coldkey", "new_hotkey", "faucet", "subnet register"):
        assert keyword in text


def test_docs_describe_the_validator_and_demo_miner_boundary():
    readme = (REPO_ROOT / "README.md").read_text().lower()
    for keyword in (
        "private problem server",
        "validator",
        "demo miner",
        "sandbox",
        "rewards",
    ):
        assert keyword in readme
