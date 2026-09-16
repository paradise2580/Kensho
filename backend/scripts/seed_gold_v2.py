#!/usr/bin/env python3
"""v2 seed for data/gold/gold_v2.jsonl — extends v1, doesn't replace it.

Same discipline as ``seed_gold_v1.py``: every query, answer, and
``relevant_parallel_ids`` value below was written by hand after reading the
actual EN and JA text of the corresponding page. v1's items are kept
verbatim (imported, not retyped) and five new topics are added, all with a
real EN/JA parallel page in the corpus:

- taint-and-toleration  - scheduling / eviction
- network-policies      - services-networking
- service-accounts      - security
- disruptions           - PodDisruptionBudget semantics
- pod-security-standards - security

``data/gold/gold_v1.jsonl`` is left untouched deliberately:
``scripts/eval_gate.py``'s CI floors are calibrated against it, and a gold
set a CI gate depends on should only grow via a new version, never be
edited in place under an existing file name.

Run once:
    python scripts/seed_gold_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.gold import GoldItem, save_gold  # noqa: E402
from seed_gold_v1 import ITEMS as V1_ITEMS  # noqa: E402

NEW_ITEMS: list[GoldItem] = [
    GoldItem(
        id="taint-toleration-en", lang="en", topic="taint-and-toleration",
        query="What are the three taint effect values a node can use?",
        relevant_parallel_ids=("docs/concepts/scheduling-eviction/taint-and-toleration.md",),
        reference_answer="NoSchedule, PreferNoSchedule, and NoExecute.",
    ),
    GoldItem(
        id="taint-toleration-ja", lang="ja", topic="taint-and-toleration",
        query="ノードのtaintに設定できる3つのeffectの値は何ですか?",
        relevant_parallel_ids=("docs/concepts/scheduling-eviction/taint-and-toleration.md",),
        reference_answer="NoSchedule、PreferNoSchedule、NoExecuteの3つです。",
    ),
    GoldItem(
        id="network-policies-en", lang="en", topic="network-policies",
        query="Before any NetworkPolicy selects a pod, is that pod isolated or "
              "non-isolated for ingress and egress traffic?",
        relevant_parallel_ids=("docs/concepts/services-networking/network-policies.md",),
        reference_answer="Non-isolated for both — by default all inbound and "
                         "outbound connections are allowed.",
    ),
    GoldItem(
        id="network-policies-ja", lang="ja", topic="network-policies",
        query="NetworkPolicyが何も存在しない名前空間では、デフォルトでPodの内向き・"
              "外向き通信はどうなりますか?",
        relevant_parallel_ids=("docs/concepts/services-networking/network-policies.md",),
        reference_answer="内向き・外向きともにすべてのトラフィックが許可されます。",
    ),
    GoldItem(
        id="service-accounts-en", lang="en", topic="service-accounts",
        query="Does every namespace automatically get a default ServiceAccount, "
              "and what permissions does it start with?",
        relevant_parallel_ids=("docs/concepts/security/service-accounts.md",),
        reference_answer="Yes — Kubernetes automatically creates a ServiceAccount "
                         "named `default` in every namespace; it gets no permissions "
                         "beyond the default API discovery permissions granted to all "
                         "authenticated principals when RBAC is enabled.",
    ),
    GoldItem(
        id="service-accounts-ja", lang="ja", topic="service-accounts",
        query="すべてのNamespaceには自動的にdefaultという名前のServiceAccountが"
              "作成されますか?",
        relevant_parallel_ids=("docs/concepts/security/service-accounts.md",),
        reference_answer="はい。Kubernetesはクラスター内の各Namespaceに対して"
                         "`default`という名前のServiceAccountオブジェクトを"
                         "自動的に作成します。",
    ),
    GoldItem(
        id="disruptions-en", lang="en", topic="disruptions",
        query="Can a PodDisruptionBudget prevent involuntary disruptions?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/disruptions.md",),
        reference_answer="No — PodDisruptionBudgets cannot prevent involuntary "
                         "disruptions; they only constrain voluntary ones.",
    ),
    GoldItem(
        id="disruptions-ja", lang="ja", topic="disruptions",
        query="PodDisruptionBudgetは非自発的なDisruptionを防ぐことができますか?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/disruptions.md",),
        reference_answer="いいえ、防げません。PDBが制約できるのは自発的な"
                         "Disruptionのみです。",
    ),
    GoldItem(
        id="pod-security-standards-en", lang="en", topic="pod-security-standards",
        query="What are the three Pod Security Standards policy levels, from "
              "least to most restrictive?",
        relevant_parallel_ids=("docs/concepts/security/pod-security-standards.md",),
        reference_answer="Privileged, Baseline, and Restricted.",
    ),
    GoldItem(
        id="pod-security-standards-ja", lang="ja", topic="pod-security-standards",
        query="Podセキュリティの標準における3つのポリシーレベルを、"
              "制限が緩い順に挙げてください。",
        relevant_parallel_ids=("docs/concepts/security/pod-security-standards.md",),
        reference_answer="特権(Privileged)、ベースライン(Baseline)、"
                         "制限(Restricted)の3つです。",
    ),
]

ITEMS: list[GoldItem] = [*V1_ITEMS, *NEW_ITEMS]


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "gold_v2.jsonl"
    save_gold(ITEMS, out_path)
    n_answerable = sum(1 for it in ITEMS if it.answerable)
    print(f"wrote {len(ITEMS)} gold items -> {out_path}  ({len(NEW_ITEMS)} new vs. v1)")
    print(f"  answerable   : {n_answerable}")
    print(f"  unanswerable : {len(ITEMS) - n_answerable}")
    print(f"  by lang      : en={sum(1 for it in ITEMS if it.lang == 'en')}  "
          f"ja={sum(1 for it in ITEMS if it.lang == 'ja')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
