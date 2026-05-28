"""历史训练记录导入工具

扫描 output/*.out 文件和 checkpoints/ 目录，
将已有的训练结果导入数据库，使其在 API 和前端可见。

用法:
    .venv\Scripts\python.exe backend/scripts/import_history.py
"""

import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal, engine, Base
from backend.db.models import (
    TrainingTask, TrainingStatus, TrainingLog, MetricRecord, Checkpoint,
)


OUTPUT_DIR = Path("output")
CHECKPOINT_DIR = Path("checkpoints")

# 正则表达式 - 从 build_results_data.py 复制
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
GIT_COMMIT_RE = re.compile(r"Git Commit:\s*(\S+)")


def parse_namespace_line(line: str) -> dict:
    """解析 Namespace(...) 行"""
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


def parse_timestamp(file_name: str) -> str:
    """从文件名解析时间戳"""
    m = TIMESTAMP_RE.search(file_name)
    if not m:
        return "1970-01-01T00:00:00"
    return f"{m.group('date')}T{m.group('h')}:{m.group('m')}:{m.group('s')}"


def parse_out_file(path: Path) -> dict | None:
    """解析单个 .out 文件"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # 提取 Namespace
    namespace = {}
    for line in lines[:10]:
        if line.startswith("Namespace("):
            namespace = parse_namespace_line(line)
            break

    gnn = str(namespace.get("gnn_type") or "").lower()
    num_domain = int(namespace.get("num_domain") or 0)
    if num_domain not in {2, 4, 8, 16} or not gnn:
        return None

    # 解析指标行
    rounds = []
    final_metric = None
    for line in lines:
        m = ROUND_LINE_RE.match(line)
        if m:
            # phase 统一为 KG/KT/FT
            phase_raw = m.group("phase")
            if phase_raw == "Fine-tuning":
                phase = "FT"
            elif phase_raw in ("GAT", "GraphSAGE", "LightGCN"):
                # 看是哪一阶段：如果有 round 0 之前是 KG，但这里需要区分
                # 通过上下文：KT 有 target domain 名字
                phase = "KT"
            else:
                phase = "KG"
            rounds.append({
                "domain": m.group("domain"),
                "phase": phase,
                "round": int(m.group("round")),
                "hr5": float(m.group("hr5")),
                "ndcg5": float(m.group("ndcg5")),
                "hr10": float(m.group("hr10")),
                "ndcg10": float(m.group("ndcg10")),
            })
            continue
        fm = FINAL_LINE_RE.match(line)
        if fm:
            final_metric = {
                "hr5": float(fm.group("hr5")),
                "ndcg5": float(fm.group("ndcg5")),
                "hr10": float(fm.group("hr10")),
                "ndcg10": float(fm.group("ndcg10")),
            }

    if not rounds:
        return None

    timestamp = parse_timestamp(path.name)
    return {
        "id": f"{gnn}_{num_domain}_{timestamp}",
        "file": str(path),
        "timestamp": timestamp,
        "namespace": namespace,
        "rounds": rounds,
        "final": final_metric,
        "all_lines": lines,
    }


def find_associated_checkpoints(namespace: dict) -> list[str]:
    """查找与训练参数关联的 checkpoint 目录"""
    gnn = namespace.get("gnn_type", "gat")
    num_domain = namespace.get("num_domain", 4)
    target_domain = namespace.get("target_domain", 1)
    dataset = namespace.get("dataset", "amazon")

    associated = []
    if not CHECKPOINT_DIR.exists():
        return associated

    for ckpt_dir in sorted(CHECKPOINT_DIR.iterdir(), reverse=True):
        if not ckpt_dir.is_dir():
            continue
        dir_name = ckpt_dir.name
        # 匹配: kt_amazon_4domains_target1_20260405_103138
        pattern = re.compile(
            rf"(kg|kt)_{dataset}_{num_domain}domains_target{target_domain}_\d+"
        )
        if pattern.match(dir_name):
            associated.append(dir_name)

    return associated


def import_history():
    """主导入函数"""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # 检查是否已有历史记录
    existing = db.query(TrainingTask).filter(
        TrainingTask.name.like("[历史]%")
    ).count()
    if existing > 0:
        print(f"已导入 {existing} 条历史记录，跳过重复导入")
        print("如需重新导入，请先清空数据库")
        db.close()
        return

    if not OUTPUT_DIR.exists():
        print(f"输出目录 {OUTPUT_DIR} 不存在")
        db.close()
        return

    out_files = sorted(OUTPUT_DIR.rglob("*.out"))
    print(f"找到 {len(out_files)} 个 .out 文件")

    imported = 0
    for path in out_files:
        parsed = parse_out_file(path)
        if not parsed:
            continue

        ns = parsed["namespace"]
        gnn = str(ns.get("gnn_type", "gat")).lower()
        num_domain = int(ns.get("num_domain", 4))
        target_domain = int(ns.get("target_domain", 1))
        timestamp_str = parsed["timestamp"]

        # 跳过归档文件
        if "archives" in str(path):
            continue

        # 创建任务
        task = TrainingTask(
            name=f"[历史] {gnn.upper()} {num_domain}domains",
            description=f"从 {path.name} 导入的历史训练记录",
            creator_id=None,
            gnn_type=gnn,
            dataset=ns.get("dataset", "amazon"),
            num_domain=num_domain,
            target_domain=target_domain,
            embedding_size=int(ns.get("embedding_size", 16)),
            round_gat=int(ns.get("round_gat", 30)),
            round_ft=int(ns.get("round_ft", 60)),
            lr_gnn=float(ns.get("lr_gnn", ns.get("lr_gat", 0.001))),
            lr_mf=float(ns.get("lr_mf", 0.005)),
            dp=bool(ns.get("dp", True)),
            eps=float(ns.get("eps", 8.0)),
            random_seed=int(ns.get("random_seed", 42)),
            local_epoch=int(ns.get("local_epoch", 3)),
            user_batch=int(ns.get("user_batch", 16)),
            weight_decay=float(ns.get("weight_decay", 1e-4)),
            num_negative=int(ns.get("num_negative", 4)),
            knowledge_gate=bool(ns.get("use_knowledge_gate", True)),
            knowledge_gate_threshold=float(ns.get("knowledge_gate_threshold", 0.5)),
            status=TrainingStatus.COMPLETED.value,
            progress=100.0,
            output_file=str(path),
            created_at=datetime.fromisoformat(timestamp_str),
            started_at=datetime.fromisoformat(timestamp_str),
            finished_at=datetime.fromisoformat(timestamp_str),
        )

        # 设置最终指标
        if parsed["final"]:
            task.best_hr5 = parsed["final"]["hr5"]
            task.best_hr10 = parsed["final"]["hr10"]
            task.best_ndcg5 = parsed["final"]["ndcg5"]
            task.best_ndcg10 = parsed["final"]["ndcg10"]

        db.add(task)
        db.flush()  # 获取 task.id

        # 创建指标记录
        step = 0
        for r in parsed["rounds"]:
            step += 1
            record = MetricRecord(
                task_id=task.id,
                step=step,
                stage=r["phase"],
                domain=r["domain"],
                round=r["round"],
                hr_5=r["hr5"],
                ndcg_5=r["ndcg5"],
                hr_10=r["hr10"],
                ndcg_10=r["ndcg10"],
            )
            db.add(record)

        # 创建日志记录（每 10 行保存一条以节省空间）
        for i, line in enumerate(parsed["all_lines"]):
            if i % 10 == 0 or line.startswith("[") or line.startswith("Error"):
                log = TrainingLog(
                    task_id=task.id,
                    level="ERROR" if line.startswith("Error") else "INFO",
                    message=line[:500],
                )
                db.add(log)

        # 关联 Checkpoint
        ckpts = find_associated_checkpoints(ns)
        if ckpts:
            task.checkpoint_paths = ckpts

        imported += 1
        best_hr10 = f"{parsed['final']['hr10']:.4f}" if parsed['final'] else "N/A"
        print(f"  ✓ #{task.id} {gnn.upper()} {num_domain}domains ({len(parsed['rounds'])} rounds, best HR@10={best_hr10})")

    db.commit()
    db.close()
    print(f"\n总计导入 {imported} 条训练记录")


if __name__ == "__main__":
    import_history()
