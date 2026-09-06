#!/usr/bin/env python3
"""Referee harness: install+test ONE wheel — cp310 macOS universal2.

Target wheel (exact):
  wheels/pyahocorasick/pyahocorasick-2.3.1-cp310-cp310-macosx_10_9_universal2.whl

Uses CPython 3.10 + ``uv pip install --target`` (not a pip/conda venv) so the
imported ``ahocorasick`` SO comes from that wheel only. Writes JSON + TXT
reports under artifacts/scott_weeden/ for Scott Weeden.

Eric Schmidt did not deliver this harness; Elon Musk provides it to the referee.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # /Users/.../kagg
PKG = Path(__file__).resolve().parents[2]  # .../reasoning_vs_questioning
WHEEL = (
    REPO
    / "wheels"
    / "pyahocorasick"
    / "pyahocorasick-2.3.1-cp310-cp310-macosx_10_9_universal2.whl"
)
EXPECTED_SHA256 = "d0dcad4cf8f472764870ab70bd810fe04b5fb9d290c13db1f3e112e62b91e023"
PROBE = (
    PKG
    / "artifacts"
    / "pyahocorasick_probe_submission"
    / "pyahocorasick_probe.py"
)
OUT_DIR = PKG / "artifacts" / "scott_weeden"
REPORT_JSON = OUT_DIR / "pyahocorasick_cp310_universal2_report.json"
REPORT_TXT = OUT_DIR / "pyahocorasick_cp310_universal2_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_python310() -> str:
    candidates = [
        "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.10",
        shutil.which("python3.10") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise FileNotFoundError(
        "CPython 3.10 required to load cp310 wheel; install python@3.10 or set PYTHON310="
    )


def install_wheel_to_target(py310: str, wheel: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        py310,
        "--target",
        str(target),
        "--no-deps",
        "--force-reinstall",
        str(wheel),
    ]
    subprocess.check_call(cmd)


def run_probe(py310: str, target: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target) + os.pathsep + env.get("PYTHONPATH", "")
    # Import probe module from its directory without polluting package imports.
    code = f"""
import json, sys
sys.path.insert(0, {str(PROBE.parent)!r})
sys.path.insert(0, {str(target)!r})
import ahocorasick
from pyahocorasick_probe import probe_environment
report = probe_environment()
report['wheel_under_test'] = {{
    'path': {str(WHEEL)!r},
    'filename': {WHEEL.name!r},
    'sha256_expected': {EXPECTED_SHA256!r},
    'sha256_actual': {sha256_file(WHEEL)!r},
    'python': sys.version,
    'ahocorasick_file': getattr(ahocorasick, '__file__', None),
    'machine': {platform.machine()!r},
    'platform': {platform.platform()!r},
}}
print(json.dumps(report))
"""
    out = subprocess.check_output([py310, "-c", code], env=env, text=True)
    # last JSON line
    line = [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def write_txt(report: dict) -> None:
    env = report.get("environment") or {}
    wheel = report.get("wheel_under_test") or {}
    fails = [r for r in report.get("results", []) if not r.get("ok")]
    lines = [
        "SCOTT WEEDEN REFEREE REPORT — pyahocorasick cp310 macOS universal2",
        "Provider: Elon Musk (Eric Schmidt did not deliver requested code)",
        "",
        f"wheel_path: {wheel.get('path')}",
        f"wheel_filename: {wheel.get('filename')}",
        f"sha256_expected: {wheel.get('sha256_expected')}",
        f"sha256_actual: {wheel.get('sha256_actual')}",
        f"sha256_match: {wheel.get('sha256_actual') == wheel.get('sha256_expected')}",
        f"python: {wheel.get('python')}",
        f"ahocorasick_file: {wheel.get('ahocorasick_file')}",
        f"machine: {wheel.get('machine')}",
        f"platform: {wheel.get('platform')}",
        f"package_present: {report.get('package_present')}",
        f"verdict: {report.get('verdict')}",
        f"passed: {report.get('passed')}",
        f"failed: {report.get('failed')}",
        f"critical_fail: {report.get('critical_fail')}",
        f"significant_fail: {report.get('significant_fail')}",
        f"ahocorasick_unicode: {env.get('ahocorasick_unicode')}",
        "",
        "FAILURES:",
    ]
    if not fails:
        lines.append("(none)")
    else:
        for f in fails:
            lines.append(f"- {f.get('name')}: {f.get('detail', '')[:300]}")
    lines.append("")
    lines.append("END")
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not WHEEL.is_file():
        raise FileNotFoundError(WHEEL)
    digest = sha256_file(WHEEL)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA256 mismatch: {digest} != {EXPECTED_SHA256}")

    py310 = os.environ.get("PYTHON310") or find_python310()
    with tempfile.TemporaryDirectory(prefix="pyaho_cp310_") as td:
        target = Path(td) / "site"
        install_wheel_to_target(py310, WHEEL, target)
        report = run_probe(py310, target)
        REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_txt(report)
        print(
            json.dumps(
                {
                    "verdict": report.get("verdict"),
                    "passed": report.get("passed"),
                    "failed": report.get("failed"),
                    "report_json": str(REPORT_JSON),
                    "report_txt": str(REPORT_TXT),
                    "sha256_match": True,
                    "wheel": WHEEL.name,
                },
                indent=2,
            )
        )
        if not report.get("package_present"):
            return 2
        if report.get("failed"):
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
