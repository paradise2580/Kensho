#!/usr/bin/env python3
"""One-time seed for data/gold/faithfulness_v1.jsonl.

Same provenance discipline as ``seed_gold_v1.py``: every ``evidence`` string
is real text copied from the actual corpus page named in its topic (the same
pages read for the phase-2 retrieval gold set). For each topic there are
three hand-written claims against that evidence — one that restates it
(``supported``), one that adds a specific fabricated detail the evidence
never mentions (``unsupported``), and one that directly negates something
the evidence states (``contradicted``). Nothing here was generated or
guessed; a v2 extension should read the real page before writing a claim
about it, the same rule seed_gold_v1.py follows.

Run once:
    python scripts/seed_faithfulness_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.faithfulness_gold import FaithfulnessItem, save_faithfulness  # noqa: E402

ITEMS: list[FaithfulnessItem] = [
    # --- pod-lifecycle ---
    FaithfulnessItem(
        id="pod-lifecycle-supported-en", lang="en", topic="pod-lifecycle",
        evidence=("Pods follow a defined lifecycle, starting in the Pending phase, "
                  "moving through Running if at least one of its primary containers "
                  "starts OK, and then through either the Succeeded or Failed phases "
                  "depending on whether any container in the Pod terminated in failure.",),
        claim="A Pod's lifecycle can end in either the Succeeded or Failed phase.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="pod-lifecycle-unsupported-en", lang="en", topic="pod-lifecycle",
        evidence=("Pods follow a defined lifecycle, starting in the Pending phase, "
                  "moving through Running if at least one of its primary containers "
                  "starts OK, and then through either the Succeeded or Failed phases.",),
        claim="A Pod's lifecycle includes a Suspended phase.",
        true_label="unsupported",
        notes="Suspended is not one of the four phases named in the evidence.",
    ),
    FaithfulnessItem(
        id="pod-lifecycle-contradicted-en", lang="en", topic="pod-lifecycle",
        evidence=("Pods follow a defined lifecycle, starting in the Pending phase.",),
        claim="A Pod's lifecycle never includes a Pending phase.",
        true_label="contradicted",
    ),
    # --- init-containers ---
    FaithfulnessItem(
        id="init-containers-supported-en", lang="en", topic="init-containers",
        evidence=("This document is about init containers: containers that run to "
                  "completion during Pod initialization.",),
        claim="Init containers run to completion during Pod initialization.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="init-containers-unsupported-en", lang="en", topic="init-containers",
        evidence=("Init containers can contain utilities or setup scripts not "
                  "present in an app image.",),
        claim="Init containers can only be written in Python.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="init-containers-contradicted-en", lang="en", topic="init-containers",
        evidence=("This document is about init containers: containers that run to "
                  "completion during Pod initialization, before app containers start.",),
        claim="Init containers continue running alongside the main application container.",
        true_label="contradicted",
        notes="That description is of a sidecar container, the opposite of "
              "an init container's defining trait.",
    ),
    # --- deployment ---
    FaithfulnessItem(
        id="deployment-supported-en", lang="en", topic="deployment",
        evidence=("You describe a desired state in a Deployment, and the Deployment "
                  "controller changes the actual state to the desired state at a "
                  "controlled rate.",),
        claim="A Deployment changes the actual state to a desired state at a controlled rate.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="deployment-unsupported-en", lang="en", topic="deployment",
        evidence=("A Deployment provides declarative updates for Pods and ReplicaSets.",),
        claim="A Deployment automatically provisions cloud load balancers.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="deployment-contradicted-en", lang="en", topic="deployment",
        evidence=("A Deployment provides declarative updates for Pods and ReplicaSets.",),
        claim="A Deployment provides imperative, not declarative, updates for Pods.",
        true_label="contradicted",
    ),
    # --- statefulset ---
    FaithfulnessItem(
        id="statefulset-supported-en", lang="en", topic="statefulset",
        evidence=("StatefulSets are valuable for applications that require stable, "
                  "persistent storage.",),
        claim="StatefulSets provide stable, persistent storage for applications that need it.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="statefulset-unsupported-en", lang="en", topic="statefulset",
        evidence=("StatefulSet is the workload API object used to manage stateful applications.",),
        claim="StatefulSets automatically encrypt all persistent storage volumes.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="statefulset-contradicted-en", lang="en", topic="statefulset",
        evidence=("StatefulSets are valuable for applications that require ordered, "
                  "graceful deployment and scaling.",),
        claim="StatefulSets perform unordered, non-graceful scaling of Pods.",
        true_label="contradicted",
    ),
    # --- daemonset ---
    FaithfulnessItem(
        id="daemonset-supported-en", lang="en", topic="daemonset",
        evidence=("Deleting a DaemonSet will clean up the Pods it created.",),
        claim="Deleting a DaemonSet cleans up the Pods it created.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="daemonset-unsupported-en", lang="en", topic="daemonset",
        evidence=("A DaemonSet ensures that all (or some) Nodes run a copy of a Pod.",),
        claim="DaemonSets require a minimum of three replicas per node.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="daemonset-contradicted-en", lang="en", topic="daemonset",
        evidence=("Deleting a DaemonSet will clean up the Pods it created.",),
        claim="Deleting a DaemonSet leaves its Pods running.",
        true_label="contradicted",
    ),
    # --- job ---
    FaithfulnessItem(
        id="job-supported-en", lang="en", topic="job",
        evidence=("Suspending a Job will delete its active Pods until the Job is "
                  "resumed again.",),
        claim="Suspending a Job deletes its active Pods until it is resumed.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="job-unsupported-en", lang="en", topic="job",
        evidence=("A Job creates one or more Pods and will continue to retry "
                  "execution of the Pods until a specified number of them "
                  "successfully terminate.",),
        claim="Jobs automatically retry on a fixed schedule of one attempt per hour.",
        true_label="unsupported",
        notes="Scheduled, repeating execution is CronJob's behaviour, not Job's.",
    ),
    FaithfulnessItem(
        id="job-contradicted-en", lang="en", topic="job",
        evidence=("Suspending a Job will delete its active Pods until the Job is "
                  "resumed again.",),
        claim="Suspending a Job leaves its active Pods running.",
        true_label="contradicted",
    ),
    # --- horizontal-pod-autoscale ---
    FaithfulnessItem(
        id="hpa-supported-en", lang="en", topic="horizontal-pod-autoscale",
        evidence=("Horizontal scaling means that the response to increased load is "
                  "to deploy more Pods.",),
        claim="Horizontal scaling responds to increased load by deploying more Pods.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="hpa-unsupported-en", lang="en", topic="horizontal-pod-autoscale",
        evidence=("In Kubernetes, a HorizontalPodAutoscaler automatically updates a "
                  "workload resource with the aim of automatically scaling capacity "
                  "to match demand.",),
        claim="HorizontalPodAutoscaler requires manually restarting the cluster to take effect.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="hpa-contradicted-en", lang="en", topic="horizontal-pod-autoscale",
        evidence=("Horizontal scaling means that the response to increased load is to "
                  "deploy more Pods. This is different from vertical scaling, which "
                  "for Kubernetes would mean assigning more resources (for example: "
                  "memory or CPU) to the Pods that are already running.",),
        claim="Horizontal scaling means assigning more CPU and memory to existing Pods.",
        true_label="contradicted",
        notes="That is the evidence's definition of vertical scaling, the concept the "
              "claim is being tested against confusing it with.",
    ),
    # --- service ---
    FaithfulnessItem(
        id="service-supported-en", lang="en", topic="service",
        evidence=("You use a Service to make that set of Pods available on the "
                  "network so that clients can interact with it.",),
        claim="A Service makes a set of Pods available on the network so "
              "clients can interact with them.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="service-unsupported-en", lang="en", topic="service",
        evidence=("A key aim of Services in Kubernetes is that you don't need to "
                  "modify your existing application to use an unfamiliar service "
                  "discovery mechanism.",),
        claim="Services require modifying application source code to enable service discovery.",
        true_label="unsupported",
        notes="This is the direct opposite of the stated aim, framed as a fabricated requirement.",
    ),
    FaithfulnessItem(
        id="service-contradicted-en", lang="en", topic="service",
        evidence=("A key aim of Services in Kubernetes is that you don't need to "
                  "modify your existing application to use an unfamiliar service "
                  "discovery mechanism.",),
        claim="A Service requires that you modify your existing application to use it.",
        true_label="contradicted",
    ),
    # --- ingress ---
    FaithfulnessItem(
        id="ingress-supported-en", lang="en", topic="ingress",
        evidence=("The Ingress API is no longer being developed, and will have no "
                  "further changes or updates made to it.",),
        claim="The Ingress API is frozen and will receive no further updates.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="ingress-unsupported-en", lang="en", topic="ingress",
        evidence=("The Ingress API is generally available, and is subject to the "
                  "stability guarantees for generally available APIs. The Kubernetes "
                  "project has no plans to remove Ingress from Kubernetes.",),
        claim="Ingress was deprecated and removed from Kubernetes in version 1.19.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="ingress-contradicted-en", lang="en", topic="ingress",
        evidence=("The Ingress API is no longer being developed, and will have no "
                  "further changes or updates made to it.",),
        claim="The Ingress API continues to receive regular feature updates.",
        true_label="contradicted",
    ),
    # --- configmap ---
    FaithfulnessItem(
        id="configmap-supported-en", lang="en", topic="configmap",
        evidence=("ConfigMap does not provide secrecy or encryption.",),
        claim="ConfigMaps do not provide secrecy or encryption.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="configmap-unsupported-en", lang="en", topic="configmap",
        evidence=("ConfigMap does not provide secrecy or encryption.",),
        claim="ConfigMaps automatically encrypt their values at rest using AES-256.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="configmap-contradicted-en", lang="en", topic="configmap",
        evidence=("If the data you want to store are confidential, use a Secret "
                  "rather than a ConfigMap.",),
        claim="ConfigMaps are the recommended way to store confidential data.",
        true_label="contradicted",
    ),
    # --- secret ---
    FaithfulnessItem(
        id="secret-supported-en", lang="en", topic="secret",
        evidence=("Because Secrets can be created independently of the Pods that use "
                  "them, there is less risk of the Secret (and its data) being "
                  "exposed during the workflow of creating, viewing, and editing Pods.",),
        claim="Creating Secrets independently of Pods reduces the risk of exposing their data.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="secret-unsupported-en", lang="en", topic="secret",
        evidence=("A Secret is an object that contains a small amount of sensitive "
                  "data such as a password, a token, or a key.",),
        claim="Secrets can only be created by cluster administrators with root access.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="secret-contradicted-en", lang="en", topic="secret",
        evidence=("Because Secrets can be created independently of the Pods that use "
                  "them, there is less risk of the Secret (and its data) being exposed.",),
        claim="Creating Secrets independently of Pods increases the risk of exposing their data.",
        true_label="contradicted",
    ),
    # --- resource-quotas ---
    FaithfulnessItem(
        id="resource-quotas-supported-en", lang="en", topic="resource-quotas",
        evidence=("A resource quota, defined by a ResourceQuota object, provides "
                  "constraints that limit aggregate resource consumption per namespace.",),
        claim="A ResourceQuota object limits aggregate resource consumption per namespace.",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="resource-quotas-unsupported-en", lang="en", topic="resource-quotas",
        evidence=("Resource quotas are a tool for administrators to address the "
                  "concern that one team could use more than its fair share of resources.",),
        claim="ResourceQuotas automatically delete Pods that exceed their CPU limit.",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="resource-quotas-contradicted-en", lang="en", topic="resource-quotas",
        evidence=("A resource quota, defined by a ResourceQuota object, provides "
                  "constraints that limit aggregate resource consumption per namespace.",),
        claim="ResourceQuota objects apply cluster-wide rather than per namespace.",
        true_label="contradicted",
    ),
    # --- Japanese slice: pod-lifecycle, deployment, daemonset, configmap ---
    FaithfulnessItem(
        id="pod-lifecycle-supported-ja", lang="ja", topic="pod-lifecycle",
        evidence=("Podは定義されたライフサイクルに従い`Pending`フェーズから始まり、少なくとも"
                 "1つのプライマリーコンテナが正常に開始した場合は`Running`を経由し、次に失敗"
                 "により終了したコンテナの有無に応じて、`Succeeded`または`Failed`フェーズを"
                 "経由します。",),
        claim="Podは`Pending`フェーズから始まります。",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="pod-lifecycle-unsupported-ja", lang="ja", topic="pod-lifecycle",
        evidence=("Podは定義されたライフサイクルに従い`Pending`フェーズから始まり、`Running`"
                 "を経由し、`Succeeded`または`Failed`フェーズを経由します。",),
        claim="Podのライフサイクルには`Suspended`フェーズが含まれます。",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="pod-lifecycle-contradicted-ja", lang="ja", topic="pod-lifecycle",
        evidence=("Podは定義されたライフサイクルに従い`Pending`フェーズから始まります。",),
        claim="Podは`Pending`フェーズから始まることはありません。",
        true_label="contradicted",
    ),
    FaithfulnessItem(
        id="deployment-supported-ja", lang="ja", topic="deployment",
        evidence=("_Deployment_ はPodとReplicaSetの宣言的なアップデート機能を提供します。",),
        claim="DeploymentはPodとReplicaSetの宣言的なアップデート機能を提供します。",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="deployment-unsupported-ja", lang="ja", topic="deployment",
        evidence=("_Deployment_ はPodとReplicaSetの宣言的なアップデート機能を提供します。",),
        claim="Deploymentはクラウドのロードバランサーを自動的にプロビジョニングします。",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="deployment-contradicted-ja", lang="ja", topic="deployment",
        evidence=("_Deployment_ はPodとReplicaSetの宣言的なアップデート機能を提供します。",),
        claim="Deploymentは宣言的ではなく命令的なアップデートを提供します。",
        true_label="contradicted",
    ),
    FaithfulnessItem(
        id="daemonset-supported-ja", lang="ja", topic="daemonset",
        evidence=("DaemonSetの削除により、DaemonSetが作成したPodもクリーンアップします。",),
        claim="DaemonSetを削除すると、それが作成したPodもクリーンアップされます。",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="daemonset-unsupported-ja", lang="ja", topic="daemonset",
        evidence=("_DaemonSet_ は全て(またはいくつか)のNodeが単一のPodのコピーを稼働させる"
                 "ことを保証します。",),
        claim="DaemonSetは各Nodeに最低3つのレプリカを必要とします。",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="daemonset-contradicted-ja", lang="ja", topic="daemonset",
        evidence=("DaemonSetの削除により、DaemonSetが作成したPodもクリーンアップします。",),
        claim="DaemonSetを削除しても、そのPodは稼働し続けます。",
        true_label="contradicted",
    ),
    FaithfulnessItem(
        id="configmap-supported-ja", lang="ja", topic="configmap",
        evidence=("ConfigMapは機密性や暗号化を提供しません。",),
        claim="ConfigMapは機密性や暗号化を提供しません。",
        true_label="supported",
    ),
    FaithfulnessItem(
        id="configmap-unsupported-ja", lang="ja", topic="configmap",
        evidence=("ConfigMapは機密性や暗号化を提供しません。",),
        claim="ConfigMapはAES-256を使用して値を自動的に暗号化します。",
        true_label="unsupported",
    ),
    FaithfulnessItem(
        id="configmap-contradicted-ja", lang="ja", topic="configmap",
        evidence=("保存したいデータが機密情報である場合は、ConfigMapの代わりにSecretを"
                 "使用してください。",),
        claim="ConfigMapは機密データを保存するために推奨される方法です。",
        true_label="contradicted",
    ),
]


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "faithfulness_v1.jsonl"
    save_faithfulness(ITEMS, out_path)
    by_label: dict[str, int] = {}
    for it in ITEMS:
        by_label[it.true_label] = by_label.get(it.true_label, 0) + 1
    print(f"wrote {len(ITEMS)} faithfulness items -> {out_path}")
    print(f"  by label : {by_label}")
    print(f"  by lang  : en={sum(1 for it in ITEMS if it.lang == 'en')}  "
          f"ja={sum(1 for it in ITEMS if it.lang == 'ja')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
