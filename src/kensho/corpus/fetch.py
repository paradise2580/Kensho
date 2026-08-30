"""Acquire the Kubernetes documentation corpus.

No scraping. ``github.com/kubernetes/website`` publishes the docs as markdown
under CC BY 4.0, with ``content/en/`` and ``content/ja/`` mirroring each other
at identical relative paths. A sparse checkout of those two directories is the
whole ingestion step, and the shared relative path *is* the parallel id — the
JA/EN alignment that cross-lingual evaluation depends on comes for free.

The clone is pinned to a commit SHA and that SHA is stamped onto every chunk,
so any result can be traced to the exact corpus that produced it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..schema import Document, Lang
from .parse import split_front_matter

REPO = "https://github.com/kubernetes/website.git"
LANG_DIRS = {"en": "content/en", "ja": "content/ja"}
DOCS_PREFIX = "docs/"
URL_BASE = "https://kubernetes.io/{lang}/docs/"

# Pages that exist for site plumbing rather than to answer questions.
SKIP_SUFFIXES = ("/_index.md",)
SKIP_PARTS = ("/includes/", "/_common-resources/", "/glossary/")

# Sections excluded by default, each for a measured reason:
#
# reference/   — 928 of the 1,370 English pages, almost all auto-generated API
#                stubs with a median body of 453 characters, and only 12 of
#                them are translated. Including them makes the corpus 68%
#                English-only boilerplate, which does two bad things: it buries
#                real answers under near-empty stubs, and it stacks ~900
#                untranslated English pages against every Japanese query, so
#                Japanese retrieval scores badly for a reason that has nothing
#                to do with the retriever. That would be a measurement
#                artefact misread as a finding.
# contribute/  — documentation about writing the documentation. A support
#                assistant answering "how do I run a Pod" should never surface
#                the docs style guide.
SKIP_PREFIXES = ("docs/reference/", "docs/contribute/", "docs/test/")


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def clone(dest: Path, ref: str | None = None) -> str:
    """Sparse-clone the two language directories. Returns the commit SHA."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--sparse",
              *(["--depth", "1"] if ref is None else []), REPO, str(dest)])
        _run(["git", "sparse-checkout", "set", *LANG_DIRS.values()], cwd=dest)
    if ref:
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest)
        _run(["git", "checkout", ref], cwd=dest)
    return _run(["git", "rev-parse", "HEAD"], cwd=dest)


@dataclass(frozen=True, slots=True)
class CorpusStats:
    en_pages: int
    ja_pages: int
    parallel_pages: int
    en_only_pages: int
    commit_sha: str


def _keep(rel: str, include_reference: bool = False) -> bool:
    if not rel.startswith(DOCS_PREFIX) or not rel.endswith(".md"):
        return False
    if any(p in "/" + rel for p in SKIP_PARTS):
        return False
    if rel.endswith(SKIP_SUFFIXES):
        return False
    skip = SKIP_PREFIXES[1:] if include_reference else SKIP_PREFIXES
    return not rel.startswith(skip)


def _rel_paths(root: Path, lang: Lang, include_reference: bool = False) -> set[str]:
    base = root / LANG_DIRS[lang]
    if not base.exists():
        return set()
    return {
        str(p.relative_to(base)).replace("\\", "/")
        for p in base.rglob("*.md")
        if _keep(str(p.relative_to(base)).replace("\\", "/"), include_reference)
    }


def load_documents(root: Path, commit_sha: str,
                   langs: tuple[Lang, ...] = ("en", "ja"),
                   include_reference: bool = False) -> Iterator[Document]:
    root = Path(root)
    for lang in langs:
        base = root / LANG_DIRS[lang]
        for rel in sorted(_rel_paths(root, lang, include_reference)):
            raw = (base / rel).read_text(encoding="utf-8", errors="replace")
            fm, _ = split_front_matter(raw)
            title = str(fm.get("title") or Path(rel).stem.replace("-", " ")).strip()
            slug = rel[len(DOCS_PREFIX):].removesuffix(".md").removesuffix("/_index")
            yield Document(
                parallel_id=rel,
                lang=lang,
                title=title,
                url=URL_BASE.format(lang=lang) + slug + "/",
                body=raw,
                commit_sha=commit_sha,
            )


def corpus_stats(root: Path, commit_sha: str,
                 include_reference: bool = False) -> CorpusStats:
    en = _rel_paths(root, "en", include_reference)
    ja = _rel_paths(root, "ja", include_reference)
    return CorpusStats(
        en_pages=len(en),
        ja_pages=len(ja),
        parallel_pages=len(en & ja),
        en_only_pages=len(en - ja),
        commit_sha=commit_sha,
    )
