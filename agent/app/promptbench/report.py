from collections import defaultdict
from typing import Any, Dict, List, Optional


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("model"), r.get("family"), r.get("variant"))
        groups[key].append(r)

    summaries = []
    for (model, family, variant), group_rows in groups.items():
        valid_rows = [r for r in group_rows if "error" not in r]
        n_errors = len(group_rows) - len(valid_rows)
        n = len(valid_rows)

        if n > 0:
            accuracy = sum(1 for r in valid_rows if r.get("correct")) / n
            parse_failure_rate = sum(1 for r in valid_rows if r.get("parse_failed")) / n
            abstention_rate = sum(1 for r in valid_rows if r.get("abstained")) / n
            mean_completion_tokens = sum(r.get("completion_tokens", 0) for r in valid_rows) / n
        else:
            accuracy = None
            parse_failure_rate = None
            abstention_rate = None
            mean_completion_tokens = None

        distinct_clusters = {r.get("cluster") for r in group_rows if "cluster" in r}
        n_clusters = len(distinct_clusters)

        loco = loco_swing_pp(valid_rows)

        summary = {
            "model": model,
            "family": family,
            "variant": variant,
            "n": n,
            "n_errors": n_errors,
            "accuracy": accuracy,
            "parse_failure_rate": parse_failure_rate,
            "abstention_rate": abstention_rate,
            "mean_completion_tokens": mean_completion_tokens,
            "n_clusters": n_clusters,
            "loco_swing_pp": loco,
            "rows": group_rows,
        }
        summaries.append(summary)

    return summaries


def accuracy_per_1k_completion_tokens(accuracy: Optional[float], mean_completion_tokens: Optional[float]) -> Optional[float]:
    if accuracy is None or mean_completion_tokens is None or mean_completion_tokens == 0:
        return None
    return (accuracy / mean_completion_tokens) * 1000.0


def loco_swing_pp(rows: List[Dict[str, Any]]) -> Optional[float]:
    valid_rows = [r for r in rows if "error" not in r]
    clusters = {r.get("cluster") for r in valid_rows if "cluster" in r}
    if len(clusters) < 2:
        return None

    total_n = len(valid_rows)
    if total_n == 0:
        return None

    baseline_acc = sum(1 for r in valid_rows if r.get("correct")) / total_n

    swings = []
    for c in clusters:
        subset = [r for r in valid_rows if r.get("cluster") != c]
        if not subset:
            continue
        sub_acc = sum(1 for r in subset if r.get("correct")) / len(subset)
        swing_pp = abs(sub_acc - baseline_acc) * 100.0
        swings.append(swing_pp)

    return max(swings) if swings else 0.0


def apply_exclusions(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for s in summaries:
        row_summary = dict(s)
        reasons = []

        parse_rate = row_summary.get("parse_failure_rate")
        if parse_rate is not None and parse_rate > 0.5:
            reasons.append("High parse failure rate")

        n_clusters = row_summary.get("n_clusters", 0)
        if n_clusters < 5:
            reasons.append("UNDERPOWERED: fewer than 5 clusters")

        loco = row_summary.get("loco_swing_pp")
        if loco is not None and loco > 10.0:
            reasons.append("UNDERPOWERED: loco swing > 10.0 pp")

        # n_clusters counts error rows, so a cell whose every call failed in
        # transport can still clear the 5-cluster rule and reach the printer with
        # None metrics. Local Ollama produced no such cell; API endpoints do.
        if row_summary.get("n", 0) == 0:
            reasons.append("NO USABLE ROWS: every cell errored in transport")

        if reasons:
            row_summary["excluded"] = True
            row_summary["exclusion_reason"] = "; ".join(reasons)
        else:
            row_summary["excluded"] = False
            row_summary["exclusion_reason"] = ""

        result.append(row_summary)
    return result


def paired_deltas(
    rows: List[Dict[str, Any]],
    variant_a: str,
    variant_b: str,
    model: str,
    family: str,
) -> List[float]:
    filtered = [
        r for r in rows
        if r.get("model") == model and r.get("family") == family and "error" not in r
    ]

    items_a: Dict[Any, bool] = {}
    items_b: Dict[Any, bool] = {}

    for r in filtered:
        v = r.get("variant")
        item_id = r.get("item_id")
        correct = bool(r.get("correct"))
        if v == variant_a:
            items_a[item_id] = correct
        elif v == variant_b:
            items_b[item_id] = correct

    common_items = set(items_a.keys()) & set(items_b.keys())
    deltas = []
    for item_id in common_items:
        val_a = 1.0 if items_a[item_id] else 0.0
        val_b = 1.0 if items_b[item_id] else 0.0
        deltas.append(val_a - val_b)

    return deltas
