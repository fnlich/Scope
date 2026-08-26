"""The sample challenges, loaded as validator requests.

`examples/sample_challenges/` holds five real problems -- two Python, three
Rust -- each with its public statement and three public cases. They are the
closest thing in this repository to what a validator actually sends, and they
are hard in the way real ones are: a page of prose with the edge cases stated
rather than shown, and a hidden suite you never see.

`examples/sample_challenges/run.py` already grades a solution you wrote by
hand. This loads the same directories as REQUESTS instead, so the miner can be
pointed at them and the answer it produces graded the same way.

## How many cases the model is shown

This is the one decision here that changes what a run means, so it is a knob
rather than a default nobody reads.

Show the model all three and grade it on all three and the result is circular:
`VerifyingSolver` repairs its answer until the public examples pass, so the
grade at the end can only agree with the check that was already made. It would
report a success it could not fail to report.

So the model is shown a subset and graded on everything. Two of three by
default. `--examples 0` shows none at all, which is not an artificial handicap:
on the run this miner was built for, no public examples shipped with any task,
and the entire repair loop was therefore dead code. That is the condition worth
measuring against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

DIRECTORY_ENV = "SOLVER_CHALLENGE_DIR"
_RELATIVE = Path("examples") / "sample_challenges"


class Challenge(NamedTuple):
    name: str
    language: str
    entrypoint: str
    statement: str
    cases: list[dict[str, Any]]

    def shown(self, count: Optional[int]) -> list[dict[str, Any]]:
        """The cases the model is allowed to see.

        Taken from the FRONT, so `--examples 1` is always the same case run to
        run. A random subset would make two runs of the same challenge
        incomparable, which is the opposite of what a regression check needs.
        """
        if count is None:
            return list(self.cases)
        return list(self.cases[: max(0, count)])


def challenge_dir(start: str | os.PathLike[str] | None = None) -> Optional[Path]:
    """Where the sample challenges live, or None.

    Searched upward rather than computed from `__file__`, for the same reason
    `.env` is: this package is run from `examples/custom_miner`, the challenges
    sit at the repository root, and an installed copy may be somewhere else
    again. ``SOLVER_CHALLENGE_DIR`` overrides it outright.
    """
    override = os.environ.get(DIRECTORY_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None
    here = Path(start or Path(__file__).resolve()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / _RELATIVE
        if candidate.is_dir():
            return candidate
    return None


def names(directory: Optional[Path] = None) -> list[str]:
    """Every challenge that has both of the files it needs, sorted."""
    root = directory or challenge_dir()
    if root is None:
        return []
    found = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "cases.json").is_file() and (path / "PROBLEM.md").is_file():
            found.append(path.name)
    return found


def load(name: str, directory: Optional[Path] = None) -> Challenge:
    """One challenge, or a message naming the ones that exist."""
    root = directory or challenge_dir()
    if root is None:
        raise SystemExit(
            f"could not find {_RELATIVE} above {Path(__file__).resolve().parent}. "
            f"Set {DIRECTORY_ENV} to point at it."
        )
    # `name` reaches this from the command line, and it is about to become a
    # path. Resolve and confirm the result is still a direct child, so `../..`
    # cannot read a cases.json from somewhere else on the disk.
    path = (root / name).resolve()
    if path.parent != root.resolve() or not (path / "cases.json").is_file():
        available = ", ".join(names(root)) or "(none found)"
        raise SystemExit(f"unknown challenge {name!r}; have {available}")
    payload = json.loads((path / "cases.json").read_text(encoding="utf-8"))
    return Challenge(
        name=name,
        language=payload["language"],
        entrypoint=payload["entrypoint"],
        statement=(path / "PROBLEM.md").read_text(encoding="utf-8").strip(),
        cases=list(payload["cases"]),
    )


def load_all(
    which: Sequence[str] | None = None, directory: Optional[Path] = None
) -> list[Challenge]:
    root = directory or challenge_dir()
    return [load(name, root) for name in (which if which is not None else names(root))]


if __name__ == "__main__":  # `python -m solvers.challenges` lists what there is
    directory = challenge_dir()
    if directory is None:
        raise SystemExit(f"no challenges found; set {DIRECTORY_ENV}")
    print(f"{directory}\n")
    for challenge_name in names(directory):
        one = load(challenge_name, directory)
        print(f"  {one.name:30} {one.language:7} {one.entrypoint:18} "
              f"{len(one.cases)} public case(s)")
