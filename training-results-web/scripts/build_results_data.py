import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
TARGET_DIR = ROOT / "training-results-web" / "data"
TARGET_FILE = TARGET_DIR / "results.json"

ROUND_LINE_RE = re.compile(
    r"^\[(?P<domain>[^\]]+?)\s+(?P<phase>GAT|GraphSAGE|LightGCN|Fine-tuning)\s+Round\s+(?P<round>\d+)\]\s+"
    r"hr_5\s*=\s*(?P<hr5>\d+\.\d+),\s*ndcg_5\s*=\s*(?P<ndcg5>\d+\.\d+),\s*"
    r"hr_10\s*=\s*(?P<hr10>\d+\.\d+),\s*ndcg_10\s*=\s*(?P<ndcg10>\d+\.\d+)"
)
FINAL_LINE_RE = re.compile(
    r"^hr_5\s*=\s*(?P<hr5>\d+\.\d+),\s*ndcg_5\s*=\s*(?P<ndcg5>\d+\.\d+),\s*"
    r"hr_10\s*=\s*(?P<hr10>\d+\.\d+),\s*ndcg_10\s*=\s*(?P<ndcg10>\d+\.\d+)"
)
TIMESTAMP_RE = re.compile(
    r"_(?P<date>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})\.out$"
)
NAMESPACE_ITEM_RE = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|[^,]+)")


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_timestamp(file_name: str) -> str:
    match = TIMESTAMP_RE.search(file_name)
    if not match:
        return "1970-01-01T00:00:00"
    return (
        f"{match.group('date')}T{match.group('h')}:"
        f"{match.group('m')}:{match.group('s')}"
    )


def parse_namespace_line(line: str) -> dict:
    if not line.startswith("Namespace("):
        return {}

    content = line[len("Namespace(") : -1]
    data = {}
    for m in NAMESPACE_ITEM_RE.finditer(content):
        key = m.group(1)
        raw = m.group(2).strip()

        if raw == "None":
            data[key] = None
        elif raw == "True":
            data[key] = True
        elif raw == "False":
            data[key] = False
        elif raw.startswith("'") and raw.endswith("'"):
            data[key] = raw[1:-1]
        else:
            try:
                if "." in raw or "e" in raw.lower():
                    data[key] = float(raw)
                else:
                    data[key] = int(raw)
            except Exception:
                data[key] = raw

    return data


def parse_out_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    namespace = {}
    for line in lines[:8]:
        if line.startswith("Namespace("):
            namespace = parse_namespace_line(line)
            break

    gnn = str(namespace.get("gnn_type") or "").lower()
    num_domain = int(namespace.get("num_domain") or 0)
    if num_domain not in {4, 8, 16} or not gnn:
        return None

    rounds = []
    final_metric = None
    for line in lines:
        m = ROUND_LINE_RE.match(line)
        if m:
            rounds.append(
                {
                    "domain": m.group("domain"),
                    "phase": m.group("phase"),
                    "round": int(m.group("round")),
                    "hr5": safe_float(m.group("hr5")),
                    "ndcg5": safe_float(m.group("ndcg5")),
                    "hr10": safe_float(m.group("hr10")),
                    "ndcg10": safe_float(m.group("ndcg10")),
                }
            )
            continue

        fm = FINAL_LINE_RE.match(line)
        if fm:
            final_metric = {
                "hr5": safe_float(fm.group("hr5")),
                "ndcg5": safe_float(fm.group("ndcg5")),
                "hr10": safe_float(fm.group("hr10")),
                "ndcg10": safe_float(fm.group("ndcg10")),
            }

    if not rounds:
        return None

    timestamp = parse_timestamp(path.name)
    return {
        "id": f"{gnn}_{num_domain}_{timestamp}",
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "timestamp": timestamp,
        "meta": {
            "dataset": namespace.get("dataset"),
            "model": namespace.get("model"),
            "gnn_type": gnn,
            "num_domain": num_domain,
            "target_domain": namespace.get("target_domain"),
            "random_seed": namespace.get("random_seed"),
            "dp": namespace.get("dp"),
            "eps": namespace.get("eps"),
            "round_gat": namespace.get("round_gat"),
            "round_ft": namespace.get("round_ft"),
            "lr_mf": namespace.get("lr_mf"),
            "lr_gnn": namespace.get("lr_gnn", namespace.get("lr_gat")),
        },
        "rounds": rounds,
        "final": final_metric,
    }


def aggregate_runs(runs):
    grouped = {}
    gnn_types = set()
    for run in runs:
        gnn = run["meta"]["gnn_type"]
        domains = str(run["meta"]["num_domain"])
        gnn_types.add(gnn)
        grouped.setdefault(gnn, {}).setdefault(domains, []).append(run)

    for gnn in grouped:
        for domains in grouped[gnn]:
            grouped[gnn][domains].sort(key=lambda item: item["timestamp"], reverse=True)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_runs": len(runs),
            "gnn_types": sorted(gnn_types),
            "domain_options": [4, 8, 16],
        },
        "grouped_runs": grouped,
        "all_runs": sorted(runs, key=lambda item: item["timestamp"], reverse=True),
    }


def main():
    runs = []
    for path in OUTPUT_DIR.rglob("*.out"):
        parsed = parse_out_file(path)
        if parsed:
            runs.append(parsed)

    data = aggregate_runs(runs)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {TARGET_FILE} with {len(runs)} parsed runs.")


if __name__ == "__main__":
    main()
