import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_docs_style.py"

SKIP_DIRS = {
    ".git",
    ".venv",
    "private",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    "__pycache__",
    ".ipynb_checkpoints",
}
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".ico",
    ".zip",
    ".gz",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".gguf",
    ".woff",
    ".woff2",
    ".so",
    ".dylib",
    ".pyc",
}

BANNED_WORDS = (
    "blazing",
    "lightning-fast",
    "seamless",
    "powerful",
    "cutting-edge",
    "state-of-the-art",
    "revolutionary",
    "effortless",
    "supercharged",
)
BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in BANNED_WORDS) + r")\b",
    re.IGNORECASE,
)

EMOJI_RANGES = (
    (0x1F1E6, 0x1F1FF),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F),
)


def _tracked_files() -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        names = [line for line in proc.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        names = []
    # This file necessarily contains the banned words, so it exempts itself.
    names = [name for name in names if name != SELF]
    if names:
        return [REPO_ROOT / name for name in names if (REPO_ROOT / name).is_file()]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if str(path.relative_to(REPO_ROOT)) != SELF:
                files.append(path)
    return sorted(files)


def _text_files() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for path in _tracked_files():
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            out.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return out


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_scan_sees_the_repo():
    names = {_relative(path) for path, _ in _text_files()}
    assert "pyproject.toml" in names, "docs style scan found no repo files"
    assert not any(name.startswith("private") for name in names)


def test_no_emoji():
    bad = []
    for path, text in _text_files():
        for lineno, line in enumerate(text.splitlines(), start=1):
            for char in line:
                code = ord(char)
                if any(lo <= code <= hi for lo, hi in EMOJI_RANGES):
                    bad.append(f"{_relative(path)}:{lineno}: emoji U+{code:04X}")
                    break
    assert not bad, "\n".join(bad)


def test_no_marketing_adjectives():
    bad = []
    for path, text in _text_files():
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = BANNED_RE.search(line)
            if match:
                bad.append(f"{_relative(path)}:{lineno}: banned word {match.group(0)!r}")
    assert not bad, "\n".join(bad)


def test_readme_length():
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    num_lines = len(readme.read_text(encoding="utf-8").splitlines())
    assert 60 <= num_lines <= 120, f"README.md:1: {num_lines} lines, expected 60 to 120"
