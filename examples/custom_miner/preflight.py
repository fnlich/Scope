"""Refuse to start anywhere but Linux, with the reason and the fix.

This example is Linux-only on purpose, and the check is here so that saying so
costs one clear line at startup instead of a build failure three layers down.

**Windows cannot run it at all.** ``bittensor-wallet`` and ``bittensor-drand``
publish manylinux and macOS wheels only — there is no Windows wheel of any
version — so ``pip install '.[chain]'`` falls back to building them from source,
which needs a Rust toolchain, and that build is where a Windows install dies.
Nothing in this example can route around that: the miner has to sign with the
hotkey, and the hotkey lives in ``bittensor-wallet``.

**macOS could technically install** the chain dependencies, but is not
supported here and is not tested: a miner is a long-lived server that has to
answer within a deadline around the clock, and the operational shape of that —
systemd, headless Firefox, a firewall in front of the axon port — is
Linux shaped. Silently half-working on a laptop is worse than a clear no.

If you are on Windows, WSL2 is genuinely Linux to Python and to Firefox and is
the intended path; ``sys.platform`` there is ``linux`` and this check passes.
"""

from __future__ import annotations

import sys

LINUX_ONLY = """\
{what} runs on Linux only, and this is {system}.

Why: bittensor-wallet and bittensor-drand ship manylinux and macOS wheels only.
On Windows pip has to build them from source through a Rust toolchain, and that
build is where the install fails — the miner cannot sign without them.

What to use instead:
  * a Linux host (any x86-64 distro with glibc 2.28+ — Ubuntu 22.04/24.04,
    Debian 12, Rocky 9); this is what a miner wants anyway, since it has to
    answer around the clock; or
  * WSL2 on Windows, which is real Linux to both Python and Firefox. Install
    into the WSL filesystem, not /mnt/c.

macOS can install the chain dependencies but is not supported or tested here.
"""


def require_linux(what: str = "This miner") -> None:
    """Stop now if this is not Linux. WSL2 counts as Linux and passes."""
    if sys.platform.startswith("linux"):
        return
    system = {"win32": "Windows", "cygwin": "Windows", "darwin": "macOS"}.get(
        sys.platform, sys.platform
    )
    raise SystemExit(LINUX_ONLY.format(what=what, system=system))
