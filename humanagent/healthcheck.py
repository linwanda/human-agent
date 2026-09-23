"""数据体检：不改库，只统计，输出一份可写进论文数据章节的报告。

按流式读取处理，几百 MB 的导出文件也不会把内存吃满。
统计维度：发言量、会话分布、群聊占比、短应答占比、时间分布、长度分布。
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from . import textutil

CONV_COL = ("会话显示名", "会话", "conversation", "chat")
TALKER_COL = ("会话", "chat", "session")
SPEAKER_COL = ("发送者显示名", "发送人", "speaker", "昵称")
SELF_COL = ("是否本人", "is_self", "is_me")
TEXT_COL = ("内容", "text", "content", "消息")
TYPE_COL = ("类型", "type")
TIME_COL = ("时间", "ts", "time", "timestamp")


def _get(row: dict[str, str], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        if key in row:
            value = (row.get(key) or "").strip()
            if value:
                return value
    return default


def _kind(talker: str, display: str) -> str:
    if talker.endswith("@chatroom") or "群" in display:
        return "群聊"
    if talker.startswith("gh_"):
        return "公众号"
    return "单聊"


def scan_file(path: Path, min_chars: int = 6) -> dict:
    """扫一个导出表，返回该文件的统计。"""
    stats = {
        "file": path.name, "parent": path.parent.name,
        "rows": 0, "self": 0, "other": 0,
        "self_text": 0, "self_short": 0, "other_text": 0, "nontext": 0,
        "conversations": defaultdict(lambda: {"total": 0, "self": 0, "kind": "", "first": "", "last": ""}),
        "monthly": Counter(), "hourly": Counter(), "lengths": [], "types": Counter(),
    }
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["rows"] += 1
            talker = _get(row, TALKER_COL)
            display = _get(row, CONV_COL) or talker
            flag = _get(row, SELF_COL)
            if flag:
                is_self = flag in {"是", "1", "true", "True", "yes"}
            else:
                is_self = _get(row, SPEAKER_COL) in {"我", "本人", "晚风"}
            content = (row.get(TEXT_COL[0]) or "").strip()
            mtype = _get(row, TYPE_COL, "未知")
            ts = _get(row, TIME_COL)

            conv = stats["conversations"][display]
            conv["total"] += 1
            conv["kind"] = _kind(talker, display)
            if ts:
                conv["first"] = min(conv["first"], ts) if conv["first"] else ts
                conv["last"] = max(conv["last"], ts) if conv["last"] else ts

            if not content:
                stats["nontext"] += 1
                stats["types"][mtype] += 1
                continue
            if is_self:
                stats["self"] += 1
                conv["self"] += 1
                stats["self_text"] += 1
                stats["types"][mtype] += 1
                if textutil.is_low_information(content, min_chars):
                    stats["self_short"] += 1
                stats["lengths"].append(len(content))
                if ts:
                    stats["monthly"][ts[:7]] += 1
                    try:
                        stats["hourly"][datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").hour] += 1
                    except ValueError:
                        pass
            else:
                stats["other"] += 1
                stats["other_text"] += 1
    stats["conversations"] = dict(stats["conversations"])
    return stats


def scan(paths: list[Path], min_chars: int = 6) -> dict:
    files = [scan_file(p, min_chars) for p in paths]
    totals = Counter()
    conversations: dict[str, dict] = {}
    monthly: Counter = Counter()
    hourly: Counter = Counter()
    lengths: list[int] = []
    types: Counter = Counter()
    for item in files:
        for key in ("rows", "self", "other", "self_text", "self_short", "other_text", "nontext"):
            totals[key] += item[key]
        monthly.update(item["monthly"])
        hourly.update(item["hourly"])
        types.update(item["types"])
        lengths.extend(item["lengths"])
        for name, conv in item["conversations"].items():
            target = conversations.setdefault(name, {"total": 0, "self": 0, "kind": conv["kind"],
                                                    "first": conv["first"], "last": conv["last"]})
            target["total"] += conv["total"]
            target["self"] += conv["self"]
            target["first"] = min(target["first"], conv["first"]) if target["first"] else conv["first"]
            target["last"] = max(target["last"], conv["last"]) if target["last"] else conv["last"]
    by_kind: Counter = Counter()
    self_by_kind: Counter = Counter()
    for conv in conversations.values():
        by_kind[conv["kind"]] += 1
        self_by_kind[conv["kind"]] += conv["self"]
    return {
        "files": [f for f in files],
        "totals": dict(totals),
        "conversations": conversations,
        "by_kind": dict(by_kind),
        "self_by_kind": dict(self_by_kind),
        "monthly": dict(sorted(monthly.items())),
        "hourly": {str(h): hourly.get(h, 0) for h in range(24)},
        "types": dict(types.most_common(15)),
        "length_stats": {
            "count": len(lengths),
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "median": round(statistics.median(lengths), 1) if lengths else 0,
            "p90": round(sorted(lengths)[int(len(lengths) * 0.9)], 1) if lengths else 0,
        },
    }


def write_report(result: dict, out_dir: Path, top: int = 25) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "数据体检.json"
    md_path = out_dir / "数据体检.md"
    convs = result["conversations"]
    ranked = sorted(convs.items(), key=lambda kv: kv[1]["self"], reverse=True)

    totals = result["totals"]
    self_text = totals.get("self_text", 0) or 1
    payload = {
        "totals": totals,
        "by_kind": result["by_kind"],
        "self_by_kind": result["self_by_kind"],
        "short_reply_ratio": round(totals.get("self_short", 0) / self_text, 4),
        "length_stats": result["length_stats"],
        "monthly": result["monthly"],
        "hourly": result["hourly"],
        "types": result["types"],
        "top_self_conversations": [
            {"name": n, **{k: v for k, v in c.items()}} for n, c in ranked[:top]
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# 聊天记录数据体检报告")
    lines.append("")
    lines.append("## 总量")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 总消息行数 | {totals.get('rows', 0):,} |")
    lines.append(f"| 本人消息（含非文本） | {totals.get('self', 0):,} |")
    lines.append(f"| 本人纯文本消息 | {totals.get('self_text', 0):,} |")
    lines.append(f"| 其中短应答（训练时应剔除） | {totals.get('self_short', 0):,} |")
    lines.append(f"| 短应答占比 | {payload['short_reply_ratio']:.1%} |")
    lines.append(f"| 对方消息 | {totals.get('other', 0):,} |")
    lines.append(f"| 非文本消息（图片/文件/链接等） | {totals.get('nontext', 0):,} |")
    lines.append(f"| 会话总数 | {len(convs):,} |")
    lines.append("")
    lines.append("## 会话类型分布")
    lines.append("")
    lines.append("| 类型 | 会话数 | 本人在此类的发言 |")
    lines.append("|---|---|---|")
    for kind, count in sorted(result["by_kind"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {kind} | {count:,} | {result['self_by_kind'].get(kind, 0):,} |")
    lines.append("")
    lines.append("## 本人发言长度")
    lines.append("")
    ls = result["length_stats"]
    lines.append(f"- 平均 {ls['mean']} 字，中位数 {ls['median']} 字，90 分位 {ls['p90']} 字")
    lines.append("")
    lines.append("## 发言最多的会话（按本人发言量）")
    lines.append("")
    lines.append("| 会话 | 类型 | 本人发言 | 总消息 | 时间跨度 |")
    lines.append("|---|---|---|---|---|")
    for name, conv in ranked[:top]:
        span = f"{conv['first'][:10]} ~ {conv['last'][:10]}" if conv["first"] else "-"
        lines.append(f"| {name} | {conv['kind']} | {conv['self']:,} | {conv['total']:,} | {span} |")
    lines.append("")
    lines.append("## 按月分布（本人发言）")
    lines.append("")
    lines.append("| 月份 | 条数 |")
    lines.append("|---|---|")
    for month, count in list(result["monthly"].items())[-36:]:
        lines.append(f"| {month} | {count:,} |")
    lines.append("")
    lines.append("## 消息类型分布")
    lines.append("")
    lines.append("| 类型 | 条数 |")
    lines.append("|---|---|")
    for mtype, count in result["types"].items():
        lines.append(f"| {mtype} | {count:,} |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
