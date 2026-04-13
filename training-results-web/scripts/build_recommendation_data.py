import json
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "checkpoints"
TARGET_DIR = ROOT / "training-results-web" / "data"
TARGET_FILE = TARGET_DIR / "recommendations.json"


def load_domain_names(dataset: str, num_domain: int):
    if dataset == "amazon":
        if num_domain == 4:
            return ["Clothing", "Books", "Movies", "CDs"]
        if num_domain == 8:
            return ["Clothing", "Books", "Home", "Electronics", "Sports", "Cell", "Movies", "CDs"]
        if num_domain == 16:
            return [
                "Clothing",
                "Books",
                "Home",
                "Electronics",
                "Sports",
                "Cell",
                "Tools",
                "CDs",
                "Movies",
                "Toys",
                "Automotive",
                "Pet",
                "Kindle",
                "Office",
                "Patio",
                "Grocery",
            ]
    if dataset == "douban":
        return ["Book", "Movie", "Music"]
    return []


def load_implicit_data(dataset: str, num_domain: int):
    if dataset == "amazon":
        path = ROOT / "data" / f"{num_domain}domains" / "implicit.json"
    elif dataset == "douban":
        path = ROOT / "data" / "douban_oldver" / "implicit.json"
    else:
        return None

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_target_domain_state(model_states: dict, target_domain_id: int, target_domain_name: str):
    direct_key = f"domain_{target_domain_id}_{target_domain_name}"
    if direct_key in model_states:
        return model_states[direct_key]

    prefix = f"domain_{target_domain_id}_"
    for key, state in model_states.items():
        if key.startswith(prefix):
            return state
    return None


def parse_iso(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime(1970, 1, 1)


def normalize_args(args: dict):
    return {
        "dataset": str(args.get("dataset", "")),
        "gnn_type": str(args.get("gnn_type", "")).lower(),
        "num_domain": int(args.get("num_domain", 0)),
        "target_domain": int(args.get("target_domain", -1)),
        "random_seed": int(args.get("random_seed", -1)),
        "dp": bool(args.get("dp", False)),
        "eps": float(args.get("eps", 0)),
    }


def scan_checkpoints():
    items = []
    if not CHECKPOINT_DIR.exists():
        return items

    for child in CHECKPOINT_DIR.iterdir():
        if not child.is_dir():
            continue

        metadata_path = child / "metadata.json"
        models_path = child / "models.pt"
        if not metadata_path.exists() or not models_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage = str(metadata.get("stage", ""))
        args = normalize_args(metadata.get("args", {}))

        item = {
            "path": child,
            "stage": stage,
            "timestamp": str(metadata.get("timestamp", "1970-01-01T00:00:00")),
            "timestamp_dt": parse_iso(str(metadata.get("timestamp", "1970-01-01T00:00:00"))),
            "metadata": metadata,
            "args": args,
        }

        if stage == "knowledge_transfer":
            item["target_domain_id"] = int(metadata.get("target_domain_id", args["target_domain"]))
            item["target_domain_name"] = str(metadata.get("target_domain_name", ""))

        items.append(item)

    return items


def match_kg_for_kt(kt_item: dict, kg_items: list):
    key = kt_item["args"]
    matches = [
        kg
        for kg in kg_items
        if kg["args"]["dataset"] == key["dataset"]
        and kg["args"]["gnn_type"] == key["gnn_type"]
        and kg["args"]["num_domain"] == key["num_domain"]
        and kg["args"]["target_domain"] == key["target_domain"]
        and kg["args"]["random_seed"] == key["random_seed"]
        and kg["args"]["dp"] == key["dp"]
        and abs(kg["args"]["eps"] - key["eps"]) < 1e-8
    ]

    if not matches:
        return None

    # Prefer nearest checkpoint before KT; otherwise use nearest by absolute time gap.
    kt_time = kt_item["timestamp_dt"]
    before = [m for m in matches if m["timestamp_dt"] <= kt_time]
    if before:
        before.sort(key=lambda m: m["timestamp_dt"], reverse=True)
        return before[0]

    matches.sort(key=lambda m: abs((m["timestamp_dt"] - kt_time).total_seconds()))
    return matches[0]


def compute_top10_items(U: torch.Tensor, V: torch.Tensor, evaluate_rows: torch.Tensor):
    if evaluate_rows.ndim != 2 or evaluate_rows.size(1) < 2:
        return []
    if evaluate_rows.size(0) % 100 != 0:
        return []

    test_user = evaluate_rows[:, 0].long()
    test_item = evaluate_rows[:, 1].long()
    candidate_items = test_item.view(-1, 100)

    scores = torch.sum(U[test_user] * V[test_item], dim=1).view(-1, 100)
    _, top_idx = torch.topk(scores, k=10, dim=1, largest=True, sorted=True)
    top_items = torch.gather(candidate_items, 1, top_idx)
    return top_items.tolist()


def build_snapshot(kt_item: dict, kg_item: dict):
    args = kt_item["args"]
    dataset = args["dataset"]
    num_domain = args["num_domain"]
    gnn_type = args["gnn_type"]
    target_domain_id = int(kt_item["target_domain_id"])

    domain_names = load_domain_names(dataset, num_domain)
    if not domain_names or target_domain_id < 0 or target_domain_id >= len(domain_names):
        return None

    target_domain_name = str(kt_item["target_domain_name"] or domain_names[target_domain_id])
    implicit_data = load_implicit_data(dataset, num_domain)
    if not implicit_data:
        return None

    kt_states = torch.load(kt_item["path"] / "models.pt", map_location="cpu")
    kg_states = torch.load(kg_item["path"] / "models.pt", map_location="cpu")

    kt_target_state = resolve_target_domain_state(kt_states, target_domain_id, target_domain_name)
    kg_target_state = resolve_target_domain_state(kg_states, target_domain_id, target_domain_name)
    if kt_target_state is None or kg_target_state is None:
        return None

    U_before = kg_target_state["U"].float().cpu()
    V_before = kg_target_state["V"].float().cpu()

    U_after = kt_target_state.get("user_embedding_with_attention")
    if U_after is None:
        U_after = kt_target_state["U"]
    U_after = U_after.float().cpu()
    V_after = kt_target_state["V"].float().cpu()

    eval_rows = torch.tensor(implicit_data["server_evaluate_data"][target_domain_id], dtype=torch.long)
    top10_before = compute_top10_items(U_before, V_before, eval_rows)
    top10_after = compute_top10_items(U_after, V_after, eval_rows)
    if not top10_before or not top10_after:
        return None

    global_to_local = {}
    for global_user_idx, mappings in enumerate(implicit_data.get("user_dic", [])):
        if isinstance(mappings, dict) and target_domain_name in mappings:
            global_to_local[str(global_user_idx)] = int(mappings[target_domain_name])

    return {
        "id": kt_item["path"].name,
        "timestamp": kt_item["timestamp"],
        "checkpoint": str(kt_item["path"].relative_to(ROOT)).replace("\\", "/"),
        "before_source": str(kg_item["path"].relative_to(ROOT)).replace("\\", "/"),
        "after_source": str(kt_item["path"].relative_to(ROOT)).replace("\\", "/"),
        "dataset": dataset,
        "gnn_type": gnn_type,
        "num_domain": num_domain,
        "target_domain_id": target_domain_id,
        "target_domain_name": target_domain_name,
        "num_target_users": len(top10_before),
        "global_to_local": global_to_local,
        "top10_before": top10_before,
        "top10_after": top10_after,
    }


def aggregate_snapshots(snapshots):
    grouped = {}
    for snap in snapshots:
        gnn = snap["gnn_type"]
        domains = str(snap["num_domain"])
        grouped.setdefault(gnn, {}).setdefault(domains, []).append(snap)

    for gnn in grouped:
        for domains in grouped[gnn]:
            grouped[gnn][domains].sort(key=lambda item: item["timestamp"], reverse=True)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_snapshots": len(snapshots),
            "gnn_types": sorted({s["gnn_type"] for s in snapshots}),
            "domain_options": sorted({s["num_domain"] for s in snapshots}),
        },
        "grouped_snapshots": grouped,
        "all_snapshots": sorted(snapshots, key=lambda item: item["timestamp"], reverse=True),
    }


def main():
    checkpoints = scan_checkpoints()
    kt_items = [c for c in checkpoints if c["stage"] == "knowledge_transfer"]
    kg_items = [c for c in checkpoints if c["stage"] == "knowledge_acquisition"]

    snapshots = []
    for kt_item in kt_items:
        kg_item = match_kg_for_kt(kt_item, kg_items)
        if kg_item is None:
            continue

        snapshot = build_snapshot(kt_item, kg_item)
        if snapshot:
            snapshots.append(snapshot)

    data = aggregate_snapshots(snapshots)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {TARGET_FILE} with {len(snapshots)} snapshots.")


if __name__ == "__main__":
    main()
