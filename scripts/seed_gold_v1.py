#!/usr/bin/env python3
"""One-time seed for data/gold/gold_v1.jsonl.

This is not a generator in the usual sense — every query, answer, and
``relevant_parallel_ids`` value below was written by hand after reading the
actual EN and JA text of the corresponding page in
``data/chunks/structural_t512_o64.jsonl``. A script is the delivery
mechanism only, chosen over hand-typed JSONL because several items contain
Japanese text and a single mis-escaped line would be easy to miss by eye.

To add a v2: copy this file, append to ITEMS, bump the version in the output
path, and read the real page text for anything new before writing its
query — the whole point of a hand-labelled set is that nothing in it is
guessed.

Run once:
    python scripts/seed_gold_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kensho.eval.gold import GoldItem, save_gold  # noqa: E402

# Each pair below (topic_en, topic_ja) was written from the *same* source
# page, read in both languages, so relevant_parallel_ids is identical across
# the pair — that identity is the whole cross-lingual premise of this
# project (see schema.py's parallel_id docstring).
ITEMS: list[GoldItem] = [
    GoldItem(
        id="pod-lifecycle-en", lang="en", topic="pod-lifecycle",
        query="What are the four phases a Pod moves through during its lifecycle?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/pod-lifecycle.md",),
        reference_answer="Pending, Running, Succeeded, and Failed.",
    ),
    GoldItem(
        id="pod-lifecycle-ja", lang="ja", topic="pod-lifecycle",
        query="Podのステータスは何によって構成されていますか?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/pod-lifecycle.md",),
        reference_answer="PodのConditionのセットで構成される。",
    ),
    GoldItem(
        id="init-containers-en", lang="en", topic="init-containers",
        query="How does an init container's lifetime differ from a sidecar "
              "container's, in terms of when each stops running?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/init-containers.md",),
        reference_answer="An init container runs to completion during Pod "
                         "initialization, before app containers start; a "
                         "sidecar container starts before the main "
                         "container and continues running alongside it.",
    ),
    GoldItem(
        id="init-containers-ja", lang="ja", topic="init-containers",
        query="Initコンテナはいつ実行され、いつ終了しますか?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/init-containers.md",),
        reference_answer="Podの初期化中に、アプリケーションコンテナの前に実行され、完了する。",
    ),
    GoldItem(
        id="sidecar-containers-en", lang="en", topic="sidecar-containers",
        query="What kinds of functionality do sidecar containers typically add, "
              "without altering the primary application's code?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/sidecar-containers.md",),
        reference_answer="Logging, monitoring, security, or data synchronization.",
    ),
    GoldItem(
        id="sidecar-containers-ja", lang="ja", topic="sidecar-containers",
        query="サイドカーコンテナは何のために使用されますか?",
        relevant_parallel_ids=("docs/concepts/workloads/pods/sidecar-containers.md",),
        reference_answer="ロギング、モニタリング、セキュリティ、データ同期などの追加サービスや"
                         "機能を、メインのアプリケーションコードを変更せずに提供するため。",
    ),
    GoldItem(
        id="deployment-en", lang="en", topic="deployment",
        query="What does a Deployment provide declarative updates for?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/deployment.md",),
        reference_answer="Pods and ReplicaSets.",
    ),
    GoldItem(
        id="deployment-ja", lang="ja", topic="deployment",
        query="Deploymentは何のための宣言的なアップデート機能を提供しますか?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/deployment.md",),
        reference_answer="PodとReplicaSet。",
    ),
    GoldItem(
        id="statefulset-en", lang="en", topic="statefulset",
        query="Name two properties StatefulSets provide that a plain set of "
              "stateless replicas does not.",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/statefulset.md",),
        reference_answer="Any two of: stable unique network identifiers, "
                         "stable persistent storage, ordered graceful "
                         "deployment/scaling, ordered automated rolling updates.",
    ),
    GoldItem(
        id="statefulset-ja", lang="ja", topic="statefulset",
        query="StatefulSetとは何を管理するためのワークロードAPIですか?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/statefulset.md",),
        reference_answer="ステートフルなアプリケーション。",
    ),
    GoldItem(
        id="daemonset-en", lang="en", topic="daemonset",
        query="What happens to the Pods a DaemonSet created when you delete that DaemonSet?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/daemonset.md",),
        reference_answer="They are cleaned up (deleted) along with the DaemonSet.",
    ),
    GoldItem(
        id="daemonset-ja", lang="ja", topic="daemonset",
        query="DaemonSetを削除すると何が起こりますか?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/daemonset.md",),
        reference_answer="DaemonSetが作成したPodもクリーンアップされる。",
    ),
    GoldItem(
        id="job-en", lang="en", topic="job",
        query="What happens to a Job's active Pods when the Job is suspended?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/job.md",),
        reference_answer="They are deleted, until the Job is resumed again.",
    ),
    GoldItem(
        id="job-ja", lang="ja", topic="job",
        query="Jobを一時停止すると、稼働しているPodはどうなりますか?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/job.md",),
        reference_answer="再開されるまで全部削除される。",
    ),
    GoldItem(
        id="cron-jobs-en", lang="en", topic="cron-jobs",
        query="What does a CronJob create on a repeating schedule?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/cron-jobs.md",),
        reference_answer="Jobs.",
    ),
    GoldItem(
        id="cron-jobs-ja", lang="ja", topic="cron-jobs",
        query="CronJobは何を作成しますか?",
        relevant_parallel_ids=("docs/concepts/workloads/controllers/cron-jobs.md",),
        reference_answer="Job。",
    ),
    GoldItem(
        id="hpa-en", lang="en", topic="horizontal-pod-autoscale",
        query="What is the difference between horizontal and vertical scaling in Kubernetes?",
        relevant_parallel_ids=("docs/concepts/workloads/autoscaling/horizontal-pod-autoscale.md",),
        reference_answer="Horizontal scaling deploys more Pods in response to "
                         "load; vertical scaling assigns more resources (CPU, "
                         "memory) to the Pods already running.",
    ),
    GoldItem(
        id="hpa-ja", lang="ja", topic="horizontal-pod-autoscale",
        query="水平スケーリングとは何を意味しますか?",
        relevant_parallel_ids=("docs/concepts/workloads/autoscaling/horizontal-pod-autoscale.md",),
        reference_answer="負荷の増加に対応するために、より多くのPodをデプロイすること。",
    ),
    GoldItem(
        id="service-en", lang="en", topic="service",
        query="Why do you need a Service if a Deployment already creates and "
              "destroys Pods dynamically?",
        relevant_parallel_ids=("docs/concepts/services-networking/service.md",),
        reference_answer="Because you don't know from moment to moment how "
                         "many Pods are healthy or what their identities are; "
                         "a Service gives clients a stable way to reach the set of Pods.",
    ),
    GoldItem(
        id="service-ja", lang="ja", topic="service",
        query="Serviceを使う主要な目的の一つは何ですか?",
        relevant_parallel_ids=("docs/concepts/services-networking/service.md",),
        reference_answer="既存のアプリケーションを改修せずにサービスディスカバリの仕組みを利用できるようにすること。",
    ),
    GoldItem(
        id="ingress-en", lang="en", topic="ingress",
        query="What does the Kubernetes project recommend using instead of "
              "Ingress, and what is the status of the Ingress API?",
        relevant_parallel_ids=("docs/concepts/services-networking/ingress.md",),
        reference_answer="Gateway is recommended instead; the Ingress API is "
                         "frozen — generally available but no longer being "
                         "developed further.",
    ),
    GoldItem(
        id="ingress-ja", lang="ja", topic="ingress",
        query="このガイドの用語集で、「エッジルーター」とは何と定義されていますか?",
        relevant_parallel_ids=("docs/concepts/services-networking/ingress.md",),
        reference_answer="クラスターでファイアウォールのポリシーを強制するルーター。",
    ),
    GoldItem(
        id="dns-pod-service-en", lang="en", topic="dns-pod-service",
        query="What component configures a Pod's DNS so its containers can "
              "look up Services by name?",
        relevant_parallel_ids=("docs/concepts/services-networking/dns-pod-service.md",),
        reference_answer="kubelet.",
    ),
    GoldItem(
        id="dns-pod-service-ja", lang="ja", topic="dns-pod-service",
        query="クライアントPodのデフォルトのDNS検索リストには何が含まれますか?",
        relevant_parallel_ids=("docs/concepts/services-networking/dns-pod-service.md",),
        reference_answer="そのPod自身の名前空間と、クラスターのデフォルトドメイン。",
    ),
    GoldItem(
        id="persistent-volumes-en", lang="en", topic="persistent-volumes",
        query="What prior topics does the Persistent Volumes documentation "
              "recommend being familiar with?",
        relevant_parallel_ids=("docs/concepts/storage/persistent-volumes.md",),
        reference_answer="Volumes, StorageClasses, and VolumeAttributesClasses.",
    ),
    GoldItem(
        id="persistent-volumes-ja", lang="ja", topic="persistent-volumes",
        query="永続ボリュームのドキュメントを読む前に、何を読んでおくことが推奨されていますか?",
        relevant_parallel_ids=("docs/concepts/storage/persistent-volumes.md",),
        reference_answer="ボリューム。",
    ),
    GoldItem(
        id="storage-classes-en", lang="en", topic="storage-classes",
        query="What does a StorageClass let cluster administrators describe?",
        relevant_parallel_ids=("docs/concepts/storage/storage-classes.md",),
        reference_answer="The different classes (e.g. quality-of-service "
                         "levels, backup policies) of storage they offer.",
    ),
    GoldItem(
        id="storage-classes-ja", lang="ja", topic="storage-classes",
        query="StorageClassを読む前に、精通していることが推奨されている2つの概念は何ですか?",
        relevant_parallel_ids=("docs/concepts/storage/storage-classes.md",),
        reference_answer="ボリュームと永続ボリューム。",
    ),
    GoldItem(
        id="volumes-en", lang="en", topic="volumes",
        query="Give two use cases for Kubernetes volumes mentioned in the docs.",
        relevant_parallel_ids=("docs/concepts/storage/volumes.md",),
        reference_answer="Any two of: populating a config file from a "
                         "ConfigMap/Secret, scratch space for a Pod, sharing "
                         "a filesystem between containers in a Pod or between "
                         "Pods, durable storage across restarts, passing configuration.",
    ),
    GoldItem(
        id="volumes-ja", lang="ja", topic="volumes",
        query="コンテナがクラッシュした場合、そのコンテナ内のファイルはどうなりますか?",
        relevant_parallel_ids=("docs/concepts/storage/volumes.md",),
        reference_answer="ファイルが失われる(kubeletはコンテナをクリーンな状態で再起動する)。",
    ),
    GoldItem(
        id="configmap-en", lang="en", topic="configmap",
        query="Why should you use a Secret instead of a ConfigMap for confidential data?",
        relevant_parallel_ids=("docs/concepts/configuration/configmap.md",),
        reference_answer="ConfigMap does not provide secrecy or encryption.",
    ),
    GoldItem(
        id="configmap-ja", lang="ja", topic="configmap",
        query="機密性の高いデータを保存したい場合、ConfigMapの代わりに何を使うべきですか?",
        relevant_parallel_ids=("docs/concepts/configuration/configmap.md",),
        reference_answer="Secret。",
    ),
    GoldItem(
        id="secret-en", lang="en", topic="secret",
        query="What is one benefit of Secrets being created independently of "
              "the Pods that use them?",
        relevant_parallel_ids=("docs/concepts/configuration/secret.md",),
        reference_answer="Less risk of the Secret (and its data) being "
                         "exposed during the workflow of creating, viewing, "
                         "and editing Pods.",
    ),
    GoldItem(
        id="secret-ja", lang="ja", topic="secret",
        query="Secretを使うことで、アプリケーションコードに何を含める必要がなくなりますか?",
        relevant_parallel_ids=("docs/concepts/configuration/secret.md",),
        reference_answer="機密データ。",
    ),
    GoldItem(
        id="resource-quotas-en", lang="en", topic="resource-quotas",
        query="What problem do resource quotas address when several teams share a cluster?",
        relevant_parallel_ids=("docs/concepts/policy/resource-quotas.md",),
        reference_answer="One team using more than its fair share of resources.",
    ),
    GoldItem(
        id="resource-quotas-ja", lang="ja", topic="resource-quotas",
        query="複数のチームがクラスターを共有するとき、リソースクォータはどのような問題に対処しますか?",
        relevant_parallel_ids=("docs/concepts/policy/resource-quotas.md",),
        reference_answer="1つのチームが公平な取り分を超えてリソースを使用してしまう問題。",
    ),
    GoldItem(
        id="limit-range-en", lang="en", topic="limit-range",
        query="By default, how are compute resource limits enforced on containers "
              "in a Kubernetes cluster?",
        relevant_parallel_ids=("docs/concepts/policy/limit-range.md",),
        reference_answer="By default, containers run with unbounded compute resources.",
    ),
    GoldItem(
        id="limit-range-ja", lang="ja", topic="limit-range",
        query="LimitRangeを利用すると、名前空間内のPodまたはコンテナに対してどのような制約を課せますか?",
        relevant_parallel_ids=("docs/concepts/policy/limit-range.md",),
        reference_answer="Podまたはコンテナごとに、計算リソースの使用量の最小値と最大値を強制する。",
    ),
    GoldItem(
        id="kube-scheduler-en", lang="en", topic="kube-scheduler",
        query="Once the scheduler discovers a Pod with no Node assigned, what "
              "is it responsible for?",
        relevant_parallel_ids=("docs/concepts/scheduling-eviction/kube-scheduler.md",),
        reference_answer="Finding the best Node for that Pod to run on.",
    ),
    GoldItem(
        id="kube-scheduler-ja", lang="ja", topic="kube-scheduler",
        query="Kubernetesにおける「スケジューリング」とは何を意味しますか?",
        relevant_parallel_ids=("docs/concepts/scheduling-eviction/kube-scheduler.md",),
        reference_answer="kubeletがPodを稼働させるために、PodをNodeに割り当てること。",
    ),
    GoldItem(
        id="namespaces-en", lang="en", topic="namespaces",
        query="Are StorageClass and Node namespaced objects in Kubernetes?",
        relevant_parallel_ids=("docs/concepts/overview/working-with-objects/namespaces.md",),
        reference_answer="No — namespace-based scoping applies only to "
                         "namespaced objects like Deployments and Services; "
                         "StorageClass, Nodes, and PersistentVolumes are cluster-wide.",
    ),
    GoldItem(
        id="namespaces-ja", lang="ja", topic="namespaces",
        query="Kubernetesにおいて、Namespaceとは何ですか?",
        relevant_parallel_ids=("docs/concepts/overview/working-with-objects/namespaces.md",),
        reference_answer="同一の物理クラスター上でサポートされる仮想クラスター。",
    ),
    GoldItem(
        id="labels-en", lang="en", topic="labels",
        query="Must a label key be unique across the whole cluster, or just "
              "within a single object?",
        relevant_parallel_ids=("docs/concepts/overview/working-with-objects/labels.md",),
        reference_answer="Unique for a given object, not cluster-wide.",
    ),
    GoldItem(
        id="labels-ja", lang="ja", topic="labels",
        query="ラベルのキーには、どのような一意性の制約がありますか?",
        relevant_parallel_ids=("docs/concepts/overview/working-with-objects/labels.md",),
        reference_answer="各キーは単一のオブジェクトに対してユニークである必要がある。",
    ),
    GoldItem(
        id="nodes-en", lang="en", topic="nodes",
        query="Name the three main components that run on a Kubernetes Node.",
        relevant_parallel_ids=("docs/concepts/architecture/nodes.md",),
        reference_answer="The kubelet, a container runtime, and kube-proxy.",
    ),
    GoldItem(
        id="nodes-ja", lang="ja", topic="nodes",
        query="1つのノード上のコンポーネントには何が含まれますか?",
        relevant_parallel_ids=("docs/concepts/architecture/nodes.md",),
        reference_answer="kubelet、コンテナランタイム、kube-proxy。",
    ),
    GoldItem(
        id="controller-en", lang="en", topic="controller",
        query="In the thermostat example of a control loop, what corresponds "
              "to the 'desired state'?",
        relevant_parallel_ids=("docs/concepts/architecture/controller.md",),
        reference_answer="The temperature you set on the thermostat.",
    ),
    GoldItem(
        id="controller-ja", lang="ja", topic="controller",
        query="制御ループの例として挙げられているサーモスタットにおいて、「現在の状態」とは何を指しますか?",
        relevant_parallel_ids=("docs/concepts/architecture/controller.md",),
        reference_answer="実際の部屋の温度。",
    ),
    # --- Unanswerable: plausible-sounding support queries with no coverage in
    # this corpus, plus one nonsensical-premise case per language. Reserved
    # for phase 5 (self-correction / refusal); excluded from phase 2/3
    # retrieval metrics by the harness.
    GoldItem(
        id="unanswerable-pricing-en", lang="en", topic="unanswerable",
        query="What is the annual cost of an enterprise Kubernetes support contract?",
        relevant_parallel_ids=(), answerable=False,
        notes="Kubernetes is open-source project documentation; it has no "
              "vendor pricing to retrieve. Reserved for phase 5.",
    ),
    GoldItem(
        id="unanswerable-nonsense-en", lang="en", topic="unanswerable",
        query="What was Kubernetes' stock price on the day of its IPO?",
        relevant_parallel_ids=(), answerable=False,
        notes="Nonsensical premise — Kubernetes is not a company and has no "
              "stock. Tests refusal on a false premise, not just missing coverage.",
    ),
    GoldItem(
        id="unanswerable-pricing-ja", lang="ja", topic="unanswerable",
        query="Kubernetesのサポート契約の年間費用はいくらですか?",
        relevant_parallel_ids=(), answerable=False,
        notes="Kubernetesはオープンソースのプロジェクトドキュメントであり、ベンダーの価格情報は含まれない。",
    ),
    GoldItem(
        id="unanswerable-nonsense-ja", lang="ja", topic="unanswerable",
        query="Podの平均睡眠時間はどれくらいですか?",
        relevant_parallel_ids=(), answerable=False,
        notes="カテゴリーエラーを含む無意味な前提の質問。誤った前提への拒否をテストする。",
    ),
]


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "gold_v1.jsonl"
    save_gold(ITEMS, out_path)
    n_answerable = sum(1 for it in ITEMS if it.answerable)
    print(f"wrote {len(ITEMS)} gold items -> {out_path}")
    print(f"  answerable   : {n_answerable}")
    print(f"  unanswerable : {len(ITEMS) - n_answerable} (reserved for phase 5)")
    print(f"  by lang      : en={sum(1 for it in ITEMS if it.lang == 'en')}  "
          f"ja={sum(1 for it in ITEMS if it.lang == 'ja')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
