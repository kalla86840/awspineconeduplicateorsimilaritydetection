"""Find exact duplicate files and near-similar text/code files.

Examples:
  python scripts/detect_similarity.py --root .
  python scripts/detect_similarity.py --threshold 0.92 --format json
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDES = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
)

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SimilarPair:
    left: str
    right: str
    similarity: float
    shared_shingles: int
    left_shingles: int
    right_shingles: int


def is_excluded(path: Path, root: Path, patterns: Iterable[str]) -> bool:
    relative = path.relative_to(root)
    parts = set(relative.parts)
    for pattern in patterns:
        if pattern in parts or fnmatch.fnmatch(str(relative), pattern):
            return True
    return False


def iter_files(root: Path, excludes: Iterable[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path, root, excludes):
            continue
        yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def can_decode_as_text(path: Path, max_bytes: int) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    if b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None


def normalize_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def shingles(tokens: list[str], width: int) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < width:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def collect_files(root: Path, excludes: Iterable[str], min_size: int) -> list[FileInfo]:
    files = []
    for path in iter_files(root, excludes):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_size:
            continue
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        files.append(FileInfo(str(path.relative_to(root)), size, digest))
    return files


def duplicate_groups(files: Iterable[FileInfo]) -> list[list[FileInfo]]:
    by_hash: dict[tuple[str, int], list[FileInfo]] = defaultdict(list)
    for file_info in files:
        by_hash[(file_info.sha256, file_info.size)].append(file_info)
    return [group for group in by_hash.values() if len(group) > 1]


def text_shingles(root: Path, files: Iterable[FileInfo], shingle_width: int, max_bytes: int) -> dict[str, set[str]]:
    result = {}
    for file_info in files:
        path = root / file_info.path
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = can_decode_as_text(path, max_bytes=max_bytes)
        if text is None:
            continue
        result[file_info.path] = shingles(normalize_tokens(text), shingle_width)
    return result


def similar_pairs(
    shingle_map: dict[str, set[str]],
    threshold: float,
    min_shared_shingles: int,
) -> list[SimilarPair]:
    inverted: dict[str, list[str]] = defaultdict(list)
    for path, file_shingles in shingle_map.items():
        for shingle in file_shingles:
            inverted[shingle].append(path)

    candidates: set[tuple[str, str]] = set()
    for paths in inverted.values():
        if len(paths) < 2:
            continue
        sorted_paths = sorted(paths)
        for index, left in enumerate(sorted_paths):
            for right in sorted_paths[index + 1 :]:
                candidates.add((left, right))

    pairs = []
    for left, right in sorted(candidates):
        left_shingles = shingle_map[left]
        right_shingles = shingle_map[right]
        shared = len(left_shingles & right_shingles)
        if shared < min_shared_shingles:
            continue
        union = len(left_shingles | right_shingles)
        similarity = shared / union if union else 0.0
        if similarity >= threshold:
            pairs.append(
                SimilarPair(
                    left=left,
                    right=right,
                    similarity=round(similarity, 4),
                    shared_shingles=shared,
                    left_shingles=len(left_shingles),
                    right_shingles=len(right_shingles),
                )
            )
    return sorted(pairs, key=lambda item: (-item.similarity, item.left, item.right))


def render_text(duplicates: list[list[FileInfo]], pairs: list[SimilarPair], limit: int) -> str:
    lines = []
    lines.append(f"Exact duplicate groups: {len(duplicates)}")
    shown_duplicates = duplicates if limit == 0 else duplicates[:limit]
    for index, group in enumerate(shown_duplicates, start=1):
        lines.append(f"\n[{index}] {group[0].size} bytes, sha256={group[0].sha256}")
        for file_info in sorted(group, key=lambda item: item.path):
            lines.append(f"  - {file_info.path}")
    if limit and len(duplicates) > limit:
        lines.append(f"\n... {len(duplicates) - limit} more exact duplicate groups hidden. Use --limit 0 to show all.")

    lines.append(f"\nNear-similar pairs: {len(pairs)}")
    shown_pairs = pairs if limit == 0 else pairs[:limit]
    for pair in shown_pairs:
        lines.append(
            f"  - {pair.similarity:.2%}: {pair.left} <-> {pair.right} "
            f"({pair.shared_shingles} shared shingles)"
        )
    if limit and len(pairs) > limit:
        lines.append(f"  ... {len(pairs) - limit} more near-similar pairs hidden. Use --limit 0 to show all.")
    return "\n".join(lines)


def write_csv(path: Path, pairs: list[SimilarPair]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["left", "right", "similarity", "shared_shingles", "left_shingles", "right_shingles"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(asdict(pair))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Directory to scan.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Near-similarity Jaccard threshold.")
    parser.add_argument("--shingle-width", type=int, default=8, help="Number of tokens in each text shingle.")
    parser.add_argument("--min-size", type=int, default=1, help="Skip files smaller than this many bytes.")
    parser.add_argument("--max-text-bytes", type=int, default=1_000_000, help="Skip text similarity for larger files.")
    parser.add_argument("--min-shared-shingles", type=int, default=3, help="Minimum shared shingles for pair scoring.")
    parser.add_argument("--limit", type=int, default=50, help="Max groups/pairs shown in text output. Use 0 for all.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Directory name or glob to exclude. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report output format.")
    parser.add_argument("--csv", type=Path, help="Optional CSV path for near-similar pairs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 2
    if not 0 <= args.threshold <= 1:
        print("--threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.shingle_width < 1:
        print("--shingle-width must be at least 1", file=sys.stderr)
        return 2

    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    files = collect_files(root, excludes, args.min_size)
    duplicates = duplicate_groups(files)
    shingle_map = text_shingles(root, files, args.shingle_width, args.max_text_bytes)
    file_by_path = {file_info.path: file_info for file_info in files}
    pairs = similar_pairs(shingle_map, args.threshold, args.min_shared_shingles)
    pairs = [
        pair
        for pair in pairs
        if file_by_path[pair.left].sha256 != file_by_path[pair.right].sha256
    ]

    if args.csv:
        write_csv(args.csv, pairs)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "root": str(root),
                    "file_count": len(files),
                    "duplicate_groups": [[asdict(item) for item in group] for group in duplicates],
                    "similar_pairs": [asdict(pair) for pair in pairs],
                },
                indent=2,
            )
        )
    else:
        print(render_text(duplicates, pairs, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
