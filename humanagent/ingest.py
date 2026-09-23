"""聊天记录导入与清洗。

支持两种输入：
    CSV  列名自动识别（中英文常见导出工具都能命中）
    JSONL 每行一个对象，字段同上

清洗规则在论文「数据构建」章节里要逐条写清楚，这里对应实现：
    1. 去重（同一内容哈希只留一条）
    2. 群聊默认剔除（多人语境噪声大）
    3. 低信息量回复打标但不删除（训练时排除，统计时要用）
    4. 记录每条自身发言前 N 条上下文，形成「情境 -> 我的回复」样本
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from . import textutil
from .config import ROOT, Settings

TIME_KEYS = ("time", "ts", "timestamp", "date", "datetime", "时间", "日期", "时间戳")
SPEAKER_KEYS = ("speaker", "sender", "from", "nickname", "name", "user",
                "发送者显示名", "发送人", "昵称", "说话人", "发送者")
TEXT_KEYS = ("text", "content", "message", "msg", "body", "内容", "消息", "文本")
CONV_KEYS = ("会话显示名", "conversation", "chat", "contact", "room", "session",
             "群名称", "联系人", "会话", "聊天")
TYPE_KEYS = ("类型", "type", "msg_type", "msgtype")
TALKER_KEYS = ("会话", "chat", "session", "talker")
SELF_KEYS = ("is_self", "is_me", "self", "mine", "from_me", "是否本人", "是我")


def _pick(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for key in keys:
        if key in lowered and str(lowered[key]).strip() != "":
            return str(lowered[key]).strip()
    return None


def _parse_epoch(ts: str | None) -> int:
    if not ts:
        return 0
    ts = ts.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(ts[:19], fmt).timestamp())
        except ValueError:
            continue
    return 0


def load_contact_scenes() -> dict[str, str]:
    """私聊：微信备注 → 大类。精确匹配。"""
    path = ROOT / "data" / "contact_scenes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_group_scenes() -> dict[str, str]:
    """群：群名 → 大类。精确匹配。"""
    path = ROOT / "data" / "group_scenes.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_DIGIT_CHATROOM = re.compile(r"^\d+@chatroom$")


def match_scene(
    conv_name: str,
    default: str = "",
    contact_scenes: dict[str, str] | None = None,
    group_scenes: dict[str, str] | None = None,
    *,
    is_group: bool = False,
) -> str:
    """私聊按人表（contact_scenes）；群按群表（group_scenes）。数字 chatroom 标无用。"""
    if contact_scenes and conv_name in contact_scenes:
        return contact_scenes[conv_name]
    if is_group:
        if group_scenes and conv_name in group_scenes:
            return group_scenes[conv_name]
        if _DIGIT_CHATROOM.fullmatch(conv_name or ""):
            return "无用"
        return default
    return default


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(iter_rows(path))


def iter_rows(path: Path):
    """流式读取，几百 MB 的导出文件也不会把内存吃满。"""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        yield from csv.DictReader(fh)


def header_of(path: Path) -> list[str]:
    for row in iter_rows(path):
        return [str(k).strip().lstrip("\ufeff") for k in row.keys()]
    return []


def _has_self_column(path: Path) -> bool:
    header = {h.lower() for h in header_of(path)}
    return any(key.lower() in header for key in SELF_KEYS)


def import_file(
    conn: sqlite3.Connection,
    settings: Settings,
    path: Path,
    *,
    self_name: str | None = None,
    include_groups: bool = False,
    mask: bool = False,
    default_scene: str = "",
    rebuild_context: bool = True,
) -> dict[str, int]:
    stats = {"read": 0, "self_messages": 0, "other_messages": 0, "inserted": 0, "skipped_short": 0,
             "skipped_group": 0, "skipped_duplicate": 0, "skipped_official": 0,
             "nontext_placeholder": 0, "convs_touched": 0}
    conv_cache: dict[str, int] = {}
    touched: set[int] = set()
    contact_scenes = load_contact_scenes()
    group_scenes = load_group_scenes()

    # 只有在既没有「是否本人」列、也没给出昵称时，才需要预扫描判断谁是本人
    speakers: dict[str, set[str]] = {}
    if not self_name and not _has_self_column(path):
        for row in iter_rows(path):
            conv = _pick(row, CONV_KEYS) or "未知会话"
            who = _pick(row, SPEAKER_KEYS)
            if who:
                speakers.setdefault(conv, set()).add(who)

    for row in iter_rows(path):
        stats["read"] += 1
        text = _pick(row, TEXT_KEYS)
        speaker = _pick(row, SPEAKER_KEYS) or ""
        conv_name = _pick(row, CONV_KEYS) or "未知会话"
        ts = _pick(row, TIME_KEYS) or ""
        mtype = _pick(row, TYPE_KEYS) or ""
        talker = _pick(row, TALKER_KEYS) or ""
        if talker.startswith("gh_"):      # 公众号推送，不是人际对话
            stats["skipped_official"] += 1
            continue
        is_group = (talker.endswith("@chatroom") or "群" in conv_name
                    or len(speakers.get(conv_name, ())) > 2)
        if is_group and not include_groups:
            stats["skipped_group"] += 1
            continue

        flag = _pick(row, SELF_KEYS)
        if flag is not None:
            is_self = flag.strip().lower() in {"1", "true", "yes", "y", "是", "self", "me"}
        elif self_name:
            is_self = speaker == self_name
        else:
            is_self = speaker in {"我", "本人", "me", "self"}
        if not text:
            # 图片/语音/文件等没有文本，用占位符保留对话节奏，训练时会被当作低信息量剔除
            text = f"[{mtype}]" if mtype else "[非文本]"
            stats["nontext_placeholder"] += 1
        stats["self_messages" if is_self else "other_messages"] += 1

        body = textutil.normalize(text)
        if mask:
            body = textutil.mask_sensitive(body)
        is_short = textutil.is_low_information(body, settings.min_self_chars)
        if is_self and is_short:
            stats["skipped_short"] += 1

        conv_id = _get_conversation(conn, conv_cache, conv_name, is_group)
        content_hash = textutil.digest(f"{conv_name}|{ts}|{body}")
        scene = match_scene(
            conv_name, default_scene, contact_scenes,
            group_scenes, is_group=is_group)
        try:
            conn.execute(
                """INSERT INTO episodes (conv_id, ts, ts_epoch, role, text, context, scene, is_short, content_hash)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (conv_id, ts, _parse_epoch(ts), "self" if is_self else "other", body, "",
                 scene, int(is_short), content_hash),
            )
        except sqlite3.IntegrityError:
            stats["skipped_duplicate"] += 1
            continue
        stats["inserted"] += 1
        touched.add(conv_id)

    conn.commit()
    stats["convs_touched"] = len(touched)
    stats["_touched"] = list(touched)  # type: ignore[assignment]
    if rebuild_context:
        finalize_import(conn, settings, conv_ids=touched)
    return stats


def finalize_import(
    conn: sqlite3.Connection,
    settings: Settings,
    conv_ids: set[int] | list[int] | None = None,
) -> dict[str, int]:
    """导入结束后重建上下文并补全文索引。只刷本次有新行的会话。"""
    ids = set(conv_ids or [])
    ctx = _rebuild_context(conn, settings.context_turns, conv_ids=ids or None)
    fts = _index_episodes(conn)
    return {"context_rows": ctx, "fts_added": fts}


def _get_conversation(conn: sqlite3.Connection, cache: dict[str, int], name: str, is_group: bool) -> int:
    if name in cache:
        return cache[name]
    cur = conn.execute("SELECT id FROM conversations WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        cache[name] = row["id"]
    else:
        cur = conn.execute("INSERT INTO conversations (name, is_group) VALUES (?,?)", (name, int(is_group)))
        cache[name] = cur.lastrowid
    return cache[name]


def _rebuild_context(
    conn: sqlite3.Connection,
    window: int,
    conv_ids: set[int] | list[int] | None = None,
) -> int:
    """按会话与时间顺序，为每条本人发言补上前面 window 条上下文。

    上下文用占位标记标出哪句是自己说的，方便模型区分说话人。
    """
    updated = 0
    if conv_ids is None:
        ids = [r["id"] for r in conn.execute("SELECT id FROM conversations")]
    else:
        ids = list(conv_ids)
    for conv_id in ids:
        rows = list(conn.execute(
            "SELECT id, role, text FROM episodes WHERE conv_id=? ORDER BY ts_epoch, id", (conv_id,)
        ))
        lines: list[str] = []
        for row in rows:
            prefix = "我" if row["role"] == "self" else "对方"
            ctx = "\n".join(lines[-window:])
            conn.execute("UPDATE episodes SET context=? WHERE id=?", (ctx, row["id"]))
            updated += 1
            lines.append(f"{prefix}：{row['text']}")
    conn.commit()
    return updated


def _index_episodes(conn: sqlite3.Connection) -> int:
    """把 episodes 同步到 FTS5 索引（增量：只补缺失的）。"""
    indexed = {r["episode_id"] for r in conn.execute("SELECT episode_id FROM episodes_fts")}
    rows = [r for r in conn.execute("SELECT id, text, context FROM episodes") if r["id"] not in indexed]
    for row in rows:
        tokens = textutil.tokenize(f"{row['text']} {row['context'] or ''}")
        conn.execute("INSERT INTO episodes_fts (tokens, episode_id) VALUES (?,?)", (tokens, row["id"]))
    conn.commit()
    return len(rows)


def auto_detect_self_name(path: Path) -> str | None:
    """从记录里猜「我」的昵称：出现频率最高且不像对方称呼的说话人。"""
    rows = read_rows(path)
    counts: dict[str, int] = {}
    for row in rows:
        name = _pick(row, SPEAKER_KEYS)
        if name:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
