"""Parse Kubernetes-website markdown into clean text with heading structure.

The Kubernetes docs are Hugo markdown, which means three things have to be
removed before the text is fit to embed:

1. YAML front matter (``---`` delimited) — carries the title, which we keep,
   and build metadata, which we do not.
2. Hugo shortcodes (``{{< note >}}`` … ``{{< /note >}}``) — the *tags* are
   markup, but the text inside them is real documentation and must survive.
   Deleting shortcode blocks wholesale silently removes most of the warnings
   and caveats in the corpus, which are exactly the passages users ask about.
3. Structural noise — HTML comments, image embeds, raw ``<br/>``.

Headings are tracked as a stack so every block of text knows the section path
it sits under. That path is later prepended to the chunk before embedding,
which is what lets a paragraph reading "The default is 30 seconds" be
retrievable by a question that names the setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)

# Shortcodes. The naive approach — delete every {{< ... >}} — is a silent
# corpus-destroying bug: `glossary_tooltip` carries the displayed technical
# term in a `text=` attribute, and there are ~2,000 of them across EN and JA.
# Deleting those tags wholesale removes precisely the nouns users search for,
# leaving sentences like "各cloud-controller-managerは複数のを実装します".
SHORTCODE = re.compile(
    r"\{\{[<%]\s*(?P<close>/)?\s*(?P<name>[A-Za-z0-9_.-]+)(?P<attrs>[^}]*?)[>%]\}\}",
    re.DOTALL,
)
ATTR = re.compile(r'(?P<key>[A-Za-z0-9_-]+)\s*=\s*"(?P<val>[^"]*)"')
BARE_ARG = re.compile(r'"([^"]*)"')


def _attrs(raw: str) -> dict[str, str]:
    return {m.group("key"): m.group("val") for m in ATTR.finditer(raw)}


def _resolve_shortcode(m: re.Match[str]) -> str:
    """Replace a shortcode with the text it renders, or nothing.

    Only content-bearing shortcodes get a replacement. Paired wrappers such as
    ``{{< note >}}`` … ``{{< /note >}}`` resolve to empty on both tags, which
    leaves the prose between them intact — that prose is where most of the
    corpus's warnings and caveats live.
    """
    name = m.group("name").lower()
    raw = m.group("attrs") or ""
    a = _attrs(raw)

    if name == "glossary_tooltip":
        # Displayed term, else humanise the term_id so the noun still appears.
        return a.get("text") or a.get("term_id", "").replace("-", " ").replace("_", " ")
    if name == "glossary_definition":
        return a.get("prepend", "")
    if name == "feature-state":
        state = a.get("state")
        ver = a.get("for_k8s_version")
        gate = a.get("feature_gate_name")
        bits = [b for b in (state, ver) if b]
        if gate and not bits:
            return f"(feature gate: {gate})"
        return f"(feature state: {', '.join(bits)})" if bits else ""
    if name == "param":
        # Site variables (e.g. the current release number). We cannot resolve
        # them without Hugo's config, and emitting the variable *name* would
        # inject nonsense like "Version version required".
        return ""
    if name == "heading":
        bare = BARE_ARG.search(raw)
        return bare.group(1) if bare else ""
    # ref / relref / code_sample / include / tab / note / caution / warning …
    # carry no inline text of their own.
    return ""

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG = re.compile(r"</?(?:br|hr|img|div|span|p|a|b|i|em|strong)\b[^>]*/?>", re.I)
ATX_HEADING = re.compile(r"\A(#{1,6})\s+(.*?)\s*#*\s*\Z")
FENCE = re.compile(r"\A\s*(```|~~~)")
HEADING_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*\Z")
MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class Block:
    """A run of body text under one heading path."""

    section_path: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    front_matter: dict
    blocks: tuple[Block, ...]

    @property
    def body_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def split_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, raw[m.end():]


def clean_inline(text: str) -> str:
    """Strip markup that carries no retrievable meaning, keep the words."""
    text = HTML_COMMENT.sub("", text)
    text = SHORTCODE.sub(_resolve_shortcode, text)
    text = IMAGE.sub("", text)
    text = MD_LINK.sub(r"\1", text)  # keep link text, drop the target
    text = HTML_TAG.sub("", text)
    return text


def parse_markdown(raw: str, fallback_title: str = "") -> ParsedPage:
    fm, body = split_front_matter(raw)
    title = str(fm.get("title") or fallback_title or "").strip()

    heading_stack: list[tuple[int, str]] = []
    blocks: list[Block] = []
    buf: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if not buf:
            return
        text = clean_inline("\n".join(buf))
        text = MULTI_BLANK.sub("\n\n", text).strip()
        buf.clear()
        if text:
            blocks.append(Block(tuple(h for _, h in heading_stack), text))

    for line in body.splitlines():
        fence_hit = FENCE.match(line)
        if fence_hit:
            marker = fence_hit.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            buf.append(line)
            continue

        if in_fence:
            # Never interpret markdown inside a code block; "# comment" is not
            # a heading, and splitting here would corrupt YAML and shell samples.
            buf.append(line)
            continue

        h = ATX_HEADING.match(line)
        if h:
            flush()
            level = len(h.group(1))
            text = HEADING_ANCHOR.sub("", clean_inline(h.group(2))).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if text:
                heading_stack.append((level, text))
            continue

        buf.append(line)

    flush()
    return ParsedPage(title=title, front_matter=fm, blocks=tuple(blocks))
