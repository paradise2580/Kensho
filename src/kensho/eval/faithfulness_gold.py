"""The hand-labelled gold set for the claim-verification comparison.

Same discipline as ``eval/gold.py``: every ``evidence`` passage is real text
copied from an actual corpus page, and every claim was written by hand after
reading it — the ``supported`` claim restates the evidence, the
``unsupported`` claim adds a specific fabricated detail the evidence never
states, and the ``contradicted`` claim directly negates something the
evidence does state. Three labelled variants per topic exist specifically so
the 4-arm comparison in ``verify/nli.py`` has a case each method should get
right and a case (contradiction) two of the four methods structurally can't.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Label = Literal["supported", "unsupported", "contradicted"]


@dataclass(frozen=True, slots=True)
class FaithfulnessItem:
    id: str
    lang: str
    topic: str
    evidence: tuple[str, ...]
    claim: str
    true_label: Label
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        return d


def load_faithfulness(path: str | Path) -> list[FaithfulnessItem]:
    items = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["evidence"] = tuple(d["evidence"])
            items.append(FaithfulnessItem(**d))
    _check_unique_ids(items)
    return items


def save_faithfulness(items: list[FaithfulnessItem], path: str | Path) -> None:
    _check_unique_ids(items)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def _check_unique_ids(items: list[FaithfulnessItem]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate faithfulness item id: {it.id!r}")
        seen.add(it.id)
