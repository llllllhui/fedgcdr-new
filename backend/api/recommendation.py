"""推荐查询 API — 读取 recommendation.json + 在线嵌入计算"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User
from backend.auth.deps import get_current_user
from backend.core.config import settings

router = APIRouter(prefix="/api/recommendations", tags=["推荐查询"])

# 缓存的推荐数据
_reco_cache = {}
_DATA_DIR = Path("training-results-web/data")


def _load_recommendations() -> dict:
    """加载 recommendations.json"""
    if "recommendations" in _reco_cache:
        return _reco_cache["recommendations"]

    path = _DATA_DIR / "recommendations.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _reco_cache["recommendations"] = data
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def _load_results() -> dict:
    """加载 results.json（训练结果摘要）"""
    if "results" in _reco_cache:
        return _reco_cache["results"]

    path = _DATA_DIR / "results.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _reco_cache["results"] = data
        return data
    except (json.JSONDecodeError, OSError):
        return {}


# ─── API ───


@router.get("/snapshots")
def list_snapshots(
    gnn_type: Optional[str] = Query(None),
    num_domain: Optional[int] = Query(None),
    _: User = Depends(get_current_user),
):
    """列出可用的推荐快照"""
    data = _load_recommendations()
    if not data:
        return {"snapshots": [], "message": "暂无推荐数据（请运行 scripts/build_recommendation_data.py）"}

    grouped = data.get("grouped_snapshots", {})
    if not grouped:
        return {"snapshots": [], "message": "推荐数据格式异常"}

    snapshots = []
    for gnn, domains in grouped.items():
        if gnn_type and gnn != gnn_type:
            continue
        for domain_count, snaps in domains.items():
            if num_domain and str(num_domain) != str(domain_count):
                continue
            for snap in snaps:
                snapshots.append({
                    "gnn_type": gnn,
                    "num_domain": domain_count,
                    "id": snap.get("id"),
                    "target_domain_name": snap.get("target_domain_name"),
                    "timestamp": snap.get("timestamp"),
                    "before_source": snap.get("before_source"),
                    "after_source": snap.get("after_source"),
                    "global_user_count": len(snap.get("global_to_local", {})),
                })

    return {"snapshots": sorted(snapshots, key=lambda x: x.get("timestamp", ""), reverse=True)}


@router.get("/top10/{snapshot_id}/{user_index}")
def get_top10(
    snapshot_id: str,
    user_index: int,
    _: User = Depends(get_current_user),
):
    """获取指定用户在指定推荐快照中的 Top10"""
    data = _load_recommendations()
    if not data:
        raise HTTPException(status_code=404, detail="暂无推荐数据")

    # 在所有快照中查找
    grouped = data.get("grouped_snapshots", {})
    for gnn, domains in grouped.items():
        for domain_count, snaps in domains.items():
            for snap in snaps:
                if snap.get("id") == snapshot_id:
                    global_to_local = snap.get("global_to_local", {})
                    local_idx = global_to_local.get(str(user_index))

                    if local_idx is None:
                        valid_users = sorted(
                            int(k) for k in global_to_local.keys()
                            if k.isdigit()
                        )
                        return {
                            "found": False,
                            "message": f"用户 {user_index} 不在目标域中",
                            "valid_users_sample": valid_users[:10],
                            "total_valid": len(valid_users),
                        }

                    before = snap.get("top10_before", [])[local_idx] if snap.get("top10_before") else []
                    after = snap.get("top10_after", [])[local_idx] if snap.get("top10_after") else []

                    return {
                        "found": True,
                        "snapshot_id": snapshot_id,
                        "global_user_index": user_index,
                        "local_user_index": local_idx,
                        "target_domain": snap.get("target_domain_name"),
                        "top10_before": before,
                        "top10_after": after,
                    }

    raise HTTPException(status_code=404, detail=f"快照 {snapshot_id} 不存在")


@router.get("/results-summary")
def get_results_summary(_: User = Depends(get_current_user)):
    """获取训练结果摘要（供看板使用）"""
    data = _load_results()
    if not data:
        return {"summary": {}, "runs": [], "message": "暂无训练结果数据"}

    # 限制响应大小，只返回必要的摘要字段
    summary = data.get("summary", {})
    runs = data.get("grouped_runs", {})

    return {
        "summary": summary,
        "available_gnn_types": summary.get("gnn_types", []),
        "available_domains": summary.get("domain_counts", []),
        "run_count": summary.get("total_runs", 0),
    }
