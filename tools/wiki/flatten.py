"""Flatten docs_src/ into a GitHub-Wiki working tree.

GitHub Wiki uses a flat namespace: every page lives at the wiki root and is
addressed by its filename. We keep `docs_src/` organised in subfolders for
sanity, and this script copies the tree out flat, also rewriting in-page
image links from `images/foo.png` (subfolder-relative) to the wiki layout.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


PAGE_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
"""Matches `[[Page-Name]]` or `[[Display text|Page-Name]]` wiki links.

GitHub Wiki / Gollum convention: when a pipe is present, the *second*
segment is the link target and the first is the display text. We mirror
that when resolving link targets below.
"""


def collect_pages(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*.md"))


def known_page_names(pages: list[Path]) -> set[str]:
    return {p.stem for p in pages}


def rewrite_image_links(text: str, page_dir_depth: int) -> str:
    """Rewrite `images/foo.png` so it resolves under the flat wiki layout.

    Sub-page files reference `images/foo.png` relative to themselves; once
    flattened, every page lives at the root, so the link works unchanged.
    Nothing to rewrite at present; we keep the hook in case we ever flatten
    a deeper tree.
    """
    del page_dir_depth
    return text


def warn_unknown_links(page: Path, text: str, known: set[str]) -> int:
    misses = 0
    for m in PAGE_LINK_RE.finditer(text):
        # Gollum: [[Display|Target]] — target is the second segment when piped.
        target = m.group(2) if m.group(2) else m.group(1)
        if target not in known:
            print(
                f"WARN: {page.name}: unresolved wiki link {m.group(0)}",
                file=sys.stderr,
            )
            misses += 1
    return misses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="docs_src/ directory")
    ap.add_argument("dst", type=Path, help="wiki working tree (cloned .wiki.git)")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Delete all *.md at the wiki root before copying. Off by default to "
        "preserve user-authored pages.",
    )
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.is_dir():
        print(f"FATAL: source dir not found: {src}", file=sys.stderr)
        return 2
    if not dst.is_dir():
        print(f"FATAL: dest dir not found: {dst}", file=sys.stderr)
        return 2

    pages = collect_pages(src)
    if not pages:
        print(f"FATAL: no markdown files under {src}", file=sys.stderr)
        return 2

    known = known_page_names(pages)

    if args.prune:
        for f in dst.glob("*.md"):
            f.unlink()

    misses = 0
    seen_names: dict[str, Path] = {}
    for page in pages:
        if page.name in seen_names:
            print(
                f"FATAL: duplicate page name {page.name} "
                f"({page} vs {seen_names[page.name]})",
                file=sys.stderr,
            )
            return 2
        seen_names[page.name] = page

        text = page.read_text()
        text = rewrite_image_links(text, len(page.relative_to(src).parts) - 1)
        misses += warn_unknown_links(page, text, known)

        (dst / page.name).write_text(text)

    images_src = src / "images"
    if images_src.is_dir():
        images_dst = dst / "images"
        if images_dst.is_dir():
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)

    print(f"copied {len(pages)} pages to {dst}")
    if misses:
        print(f"({misses} unresolved [[wiki-link]] targets — see warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
