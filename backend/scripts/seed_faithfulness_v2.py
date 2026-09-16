#!/usr/bin/env python3
"""v2 seed for data/gold/faithfulness_v2.jsonl — extends v1, doesn't replace it.

Same provenance discipline as ``seed_faithfulness_v1.py``: every ``evidence``
string is real text copied from the actual corpus page, read before writing
each claim. v1's items are kept verbatim (imported, not retyped); five new
topics are added in English (the same five new pages seeded into
``gold_v2.jsonl``), and one of them (taint-and-toleration) also gets a
Japanese variant, to keep roughly v1's ~3:1 EN:JA ratio while still growing
JA coverage.

``data/gold/faithfulness_v1.jsonl`` is left untouched for the same reason
``gold_v1.jsonl`` is: it's what the CI eval gate's lexical-accuracy floor is
calibrated against.

Run once:
    python scripts/seed_faithfulness_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.faithfulness_gold import FaithfulnessItem, save_faithfulness  # noqa: E402
from seed_faithfulness_v1 import ITEMS as V1_ITEMS  # noqa: E402

NEW_ITEMS: list[FaithfulnessItem] = [
    # --- taint-and-toleration ---
    FaithfulnessItem(
        id="taint-toleration-supported-en", lang="en", topic="taint-and-toleration",
        evidence=("`NoSchedule`: No new Pods will be scheduled on the tainted node "
                  "unless they have a matching toleration. Pods currently running "
                  "on the node are not evicted.",),
        claim="A `NoSchedule` taint does not evict Pods already running on the node.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="taint-toleration-unsupported-en", lang="en", topic="taint-and-toleration",
        evidence=("`NoSchedule`: No new Pods will be scheduled on the tainted node "
                  "unless they have a matching toleration. Pods currently running "
                  "on the node are not evicted.",),
        claim="A `NoSchedule` taint blocks new Pods for a fixed five-minute cooldown period.",
        true_label="unsupported",
        notes="No time-bound cooldown is mentioned anywhere in the evidence.",
    ),
    FaithfulnessItem(
        id="taint-toleration-contradicted-en", lang="en", topic="taint-and-toleration",
        evidence=("`NoSchedule`: No new Pods will be scheduled on the tainted node "
                  "unless they have a matching toleration. Pods currently running "
                  "on the node are not evicted.",),
        claim="A `NoSchedule` taint immediately evicts Pods already running on the node.",
        true_label="contradicted",
        notes="Directly negates 'Pods currently running on the node are not evicted.'",
    ),
    FaithfulnessItem(
        id="taint-toleration-supported-ja", lang="ja", topic="taint-and-toleration",
        evidence=("effect `NoExecute`のtaintが残った場合、既に稼働中のPodはそのノード"
                  "から排除され、まだ稼働していないPodはスケジューリングされないよう"
                  "になります。",),
        claim="`NoExecute`のtaintが残っている場合、既に稼働中のPodはそのノードから"
              "排除されます。",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="taint-toleration-unsupported-ja", lang="ja", topic="taint-and-toleration",
        evidence=("effect `NoExecute`のtaintが残った場合、既に稼働中のPodはそのノード"
                  "から排除され、まだ稼働していないPodはスケジューリングされないよう"
                  "になります。",),
        claim="`NoExecute`のtaintが残っている場合、Podの排除には必ず30秒の猶予期間が"
              "設けられます。",
        true_label="unsupported",
        notes="猶予期間の長さについて、この根拠テキストは何も述べていない。",
    ),
    FaithfulnessItem(
        id="taint-toleration-contradicted-ja", lang="ja", topic="taint-and-toleration",
        evidence=("effect `NoExecute`のtaintが残った場合、既に稼働中のPodはそのノード"
                  "から排除され、まだ稼働していないPodはスケジューリングされないよう"
                  "になります。",),
        claim="`NoExecute`のtaintが残っていても、既に稼働中のPodはそのノードに"
              "留まり続けます。",
        true_label="contradicted",
        notes="「既に稼働中のPodはそのノードから排除され」を直接否定している。",
    ),
    # --- network-policies ---
    FaithfulnessItem(
        id="network-policies-supported-en", lang="en", topic="network-policies",
        evidence=("By default, a pod is non-isolated for ingress; all inbound "
                  "connections are allowed.",),
        claim="Without any NetworkPolicy applied, a pod allows all inbound connections "
              "by default.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="network-policies-unsupported-en", lang="en", topic="network-policies",
        evidence=("By default, a pod is non-isolated for ingress; all inbound "
                  "connections are allowed.",),
        claim="By default, a pod allows inbound connections only from pods in the "
              "same namespace.",
        true_label="unsupported",
        notes="The evidence says all inbound connections are allowed, not "
              "namespace-scoped ones specifically.",
    ),
    FaithfulnessItem(
        id="network-policies-contradicted-en", lang="en", topic="network-policies",
        evidence=("By default, a pod is non-isolated for ingress; all inbound "
                  "connections are allowed.",),
        claim="By default, a pod blocks all inbound connections until a NetworkPolicy "
              "explicitly allows them.",
        true_label="contradicted",
        notes="Directly negates 'non-isolated for ingress; all inbound connections "
              "are allowed.'",
    ),
    # --- service-accounts ---
    FaithfulnessItem(
        id="service-accounts-supported-en", lang="en", topic="service-accounts",
        evidence=("When you create a cluster, Kubernetes automatically creates a "
                  "ServiceAccount object named `default` for every namespace in your "
                  "cluster. The `default` service accounts in each namespace get no "
                  "permissions by default other than the default API discovery "
                  "permissions.",),
        claim="Kubernetes automatically creates a ServiceAccount named `default` in "
              "every namespace.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="service-accounts-unsupported-en", lang="en", topic="service-accounts",
        evidence=("When you create a cluster, Kubernetes automatically creates a "
                  "ServiceAccount object named `default` for every namespace in your "
                  "cluster. The `default` service accounts in each namespace get no "
                  "permissions by default other than the default API discovery "
                  "permissions.",),
        claim="The default ServiceAccount in each namespace is granted cluster-admin "
              "permissions automatically.",
        true_label="unsupported",
        notes="The evidence says the opposite in effect (no permissions beyond API "
              "discovery), but the claim names a specific permission level "
              "(cluster-admin) the evidence never mentions granting.",
    ),
    FaithfulnessItem(
        id="service-accounts-contradicted-en", lang="en", topic="service-accounts",
        evidence=("When you create a cluster, Kubernetes automatically creates a "
                  "ServiceAccount object named `default` for every namespace in your "
                  "cluster.",),
        claim="Namespaces do not get a default ServiceAccount unless an administrator "
              "creates one manually.",
        true_label="contradicted",
        notes="Directly negates 'Kubernetes automatically creates a ServiceAccount "
              "... for every namespace.'",
    ),
    # --- disruptions (PodDisruptionBudget) ---
    FaithfulnessItem(
        id="disruptions-supported-en", lang="en", topic="disruptions",
        evidence=("Involuntary disruptions cannot be prevented by PDBs; however "
                  "they do count against the budget.",),
        claim="Involuntary disruptions still count against a PodDisruptionBudget's "
              "budget, even though a PDB can't prevent them.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="disruptions-unsupported-en", lang="en", topic="disruptions",
        evidence=("Involuntary disruptions cannot be prevented by PDBs; however "
                  "they do count against the budget.",),
        claim="A PodDisruptionBudget's eviction requests are retried for up to "
              "24 hours before giving up.",
        true_label="unsupported",
        notes="No retry duration is stated anywhere in the evidence.",
    ),
    FaithfulnessItem(
        id="disruptions-contradicted-en", lang="en", topic="disruptions",
        evidence=("Involuntary disruptions cannot be prevented by PDBs; however "
                  "they do count against the budget.",),
        claim="Involuntary disruptions are never counted against a "
              "PodDisruptionBudget's budget.",
        true_label="contradicted",
        notes="Directly negates 'they do count against the budget.'",
    ),
    # --- pod-security-standards ---
    FaithfulnessItem(
        id="pod-security-standards-supported-en", lang="en", topic="pod-security-standards",
        evidence=("Restricted: Heavily restricted policy, following current Pod "
                  "hardening best practices.",
                  "Privileged: Unrestricted policy, providing the widest possible "
                  "level of permissions.",),
        claim="The Restricted policy follows current Pod hardening best practices.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="pod-security-standards-unsupported-en", lang="en", topic="pod-security-standards",
        evidence=("Baseline: Minimally restrictive policy which prevents known "
                  "privilege escalations. Allows the default (minimally specified) "
                  "Pod configuration.",),
        claim="The Baseline policy requires every container to explicitly set a "
              "non-root user ID.",
        true_label="unsupported",
        notes="The evidence says Baseline allows the default, minimally specified "
              "Pod configuration — it never states a mandatory non-root user ID.",
    ),
    FaithfulnessItem(
        id="pod-security-standards-contradicted-en", lang="en", topic="pod-security-standards",
        evidence=("Privileged: Unrestricted policy, providing the widest possible "
                  "level of permissions. This policy allows for known privilege "
                  "escalations.",),
        claim="The Privileged policy is the most heavily restricted of the three "
              "Pod Security Standards levels.",
        true_label="contradicted",
        notes="Directly negates 'Unrestricted policy, providing the widest possible "
              "level of permissions.'",
    ),
]

ITEMS: list[FaithfulnessItem] = [*V1_ITEMS, *NEW_ITEMS]


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "faithfulness_v2.jsonl"
    save_faithfulness(ITEMS, out_path)
    by_label = {label: sum(1 for it in ITEMS if it.true_label == label)
               for label in ("supported", "unsupported", "contradicted")}
    print(f"wrote {len(ITEMS)} faithfulness items -> {out_path}  "
         f"({len(NEW_ITEMS)} new vs. v1)")
    print(f"  by label : {by_label}")
    print(f"  by lang  : en={sum(1 for it in ITEMS if it.lang == 'en')}  "
         f"ja={sum(1 for it in ITEMS if it.lang == 'ja')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
