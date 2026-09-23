"""知识库：写入、认人、取记忆、说话方式档。

一次回复用到的三样东西，各自来源固定：
    说话方式  data/style_profile.json 的三档（熟人 / 同龄不熟 / 年长），
              每档写死长短、标点、表情、口头语和真实原话样例，不随话题漂移
    认人      data/relations.json 近圈薄卡 → data/contact_scenes.json 关系大类 → 不熟
    记忆      关系卡 + 关键词命中的事实（facts_fts），再加每次都带的人设卡
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import difflib
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import textutil
from .config import ROOT, Settings
from .llm import LLM

# 相似度 / 时间新鲜度 / 质量分 的权重
RETRIEVAL_WEIGHTS = {"similarity": 0.6, "recency": 0.25, "quality": 0.15}


def batch_map(fn, items, workers: int = 1):
    """并发执行一批请求；workers <= 1 时退化为顺序执行。"""
    if workers and workers > 1 and items:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(fn, items)
    else:
        for item in items:
            yield fn(item)

# 单值谓词：同一主语下只保留最新一条，旧的置为 superseded
FUNCTIONAL_PREDICATES = {"就读院校", "专业", "年级", "所在地", "职业", "正在做", "作息", "联系方式", "感情"}

# 这轮话里没碰到这些词，就别让全文检索把这些敏感条目捞出来。
# 例：有人自称「天线宝宝」，不该把「感情：只认真谈过宝宝」翻出来。
GATED_PREDICATES: dict[str, tuple[str, ...]] = {
    "感情": ("对象", "女友", "男友", "女朋友", "男朋友", "单身", "恋爱", "分手", "谈恋爱", "相亲", "喜欢谁"),
}

EXTRACT_PROMPT = """你是个人信息抽取器。下面是从聊天记录里剪出来的一段对话。
请只抽取关于「我」本人的信息，不要抽取对方的信息，也不要推断。

对话：
{text}

只输出 JSON，格式：
{{"facts": [{{"subject": "我", "predicate": "学校|专业|年级|所在地|实习单位|岗位|经历|技能|项目|兴趣|关系|其他",
   "object": "具体内容", "confidence": 0.0-1.0}}],
 "preferences": [{{"domain": "学习|饮食|社交|娱乐|作息|其他", "statement": "我喜欢在晚上写代码",
   "polarity": "like|dislike|neutral", "confidence": 0.0-1.0}}]}}

只抽「长期有效」的信息，规则：
1. 不要抽一时一刻的动作或念头，比如「打算询问某东西叫什么」「看视频」「考虑要不要买」，
   这类内容一律不写。
2. object 必须是名词短语（长期身份、技术栈、爱好、项目），不是一句话。
3. 对方的偏好、别人的事情一律不要抽。
4. 一段对话最多抽 3 条，没有可用信息就返回 {{"facts": [], "preferences": []}}。
5. 只有对话里明确出现过的才写，宁缺勿滥。"""

# object 以这些词开头，说明是临时动作而不是长期事实
TRANSIENT_PREFIX = ("打算", "询问", "想问", "准备问", "考虑", "看", "关注", "了解", "知道",
                    "不知道", "觉得", "认为", "想", "要", "准备")


def _is_noise_fact(item: dict[str, Any]) -> bool:
    obj = textutil.normalize(str(item.get("object", "")))
    if not (2 <= len(obj) <= 40):
        return True
    if textutil.contains_sensitive(obj):
        return True          # 密码、身份证、手机号这类一律不入库
    if obj.startswith(TRANSIENT_PREFIX):
        return True
    return float(item.get("confidence", 0.7)) < 0.7


# ---------------- 写入 ----------------

def extract_facts(
    llm: LLM,
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    batch: int = 20,
    workers: int = 4,
    max_cost: float | None = None,
) -> dict[str, int]:
    """批量调用模型，把情节记忆转成事实与偏好。

    并发只发生在 HTTP 请求上，数据库写入仍留在主线程，避免 SQLite 争用。
    max_cost 是本次运行的美元预算上限，超过就停下（不再继续提交新请求）。
    """
    sql = """SELECT e.id, e.text, e.context FROM episodes e
             LEFT JOIN extract_log l ON l.episode_id = e.id
             WHERE l.episode_id IS NULL AND e.is_short = 0
             ORDER BY e.ts_epoch DESC LIMIT ?"""
    rows = conn.execute(sql, (limit or batch,)).fetchall()
    stats = {"scanned": 0, "facts": 0, "preferences": 0, "skipped_noise": 0, "failed": 0,
             "stopped_by_budget": 0}

    def work(row: sqlite3.Row) -> tuple[int, dict | None, str]:
        # 送模型之前先脱敏：密码、身份证、手机号、邮箱这类内容不离开本机
        block = textutil.redact(f"{row['context'] or ''}\n我：{row['text']}")
        try:
            data = llm.chat_json(
                [{"role": "user", "content": EXTRACT_PROMPT.format(text=block)}],
                task="extract",
                temperature=0.0,
                thinking=False,      # 抽取是结构化任务，关掉思维链
            )
            return row["id"], data, ""
        except Exception as exc:  # 单条失败不影响整批
            return row["id"], None, str(exc)

    # 分批提交：这样每批结束后都能检查预算，超了就立刻停
    chunk = max(1, workers * 4)
    for start in range(0, len(rows), chunk):
        if max_cost is not None and llm.session_cost >= max_cost:
            stats["stopped_by_budget"] = 1
            print(f"  [预算闸] 已花费 ${llm.session_cost:.4f}，达到上限 ${max_cost}，停止。")
            break
        for episode_id, data, error in batch_map(work, rows[start:start + chunk], workers):
            stats["scanned"] += 1
            if error:
                stats["failed"] += 1
                print(f"  [警告] 第 {episode_id} 条抽取失败：{error[:120]}")
                continue
            raw_facts = (data or {}).get("facts", [])
            clean = [item for item in raw_facts if not _is_noise_fact(item)]
            stats["skipped_noise"] += len(raw_facts) - len(clean)
            stats["facts"] += upsert_facts(conn, clean, evidence=str(episode_id))
            stats["preferences"] += upsert_preferences(conn, (data or {}).get("preferences", []),
                                                       evidence=str(episode_id))
            conn.execute("INSERT OR REPLACE INTO extract_log (episode_id) VALUES (?)", (episode_id,))
            conn.commit()
    stats["cost_usd"] = round(llm.session_cost, 6)  # type: ignore[assignment]
    return stats


def upsert_facts(conn: sqlite3.Connection, facts: list[dict[str, Any]], evidence: str = "") -> int:
    added = 0
    known = _facts_by_predicate(conn)
    for item in facts:
        obj = textutil.normalize(str(item.get("object", "")))
        pred = str(item.get("predicate", "")).strip()
        if not obj or not pred:
            continue
        if any(_too_similar(obj, old) for old in known.get(pred, ())):
            continue          # 同一谓词下已有高度相似的值，不重复入库
        cur = conn.execute(
            """INSERT INTO facts (subject, predicate, object, confidence, evidence, status, valid_from)
               VALUES (?,?,?,?,?, 'active', ?)
               ON CONFLICT(subject, predicate, object) DO UPDATE SET
                 confidence = MAX(confidence, excluded.confidence),
                 evidence = COALESCE(facts.evidence,'') || ',' || excluded.evidence,
                 status = 'active'""",
            (item.get("subject", "我"), pred, obj, float(item.get("confidence", 0.7)), evidence,
             time.strftime("%Y-%m-%d")),
        )
        if cur.rowcount:
            added += 1
            known.setdefault(pred, set()).add(obj)
        if pred in FUNCTIONAL_PREDICATES:  # 单值谓词：同谓词下旧值作废
            conn.execute(
                "UPDATE facts SET status='superseded', valid_to=? WHERE predicate=? AND object<>? AND status='active'",
                (time.strftime("%Y-%m-%d"), pred, obj),
            )
    conn.commit()
    _reindex_facts(conn)
    return added


def _facts_by_predicate(conn: sqlite3.Connection) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in conn.execute("SELECT predicate, object FROM facts WHERE status='active'"):
        out.setdefault(row["predicate"], set()).add(row["object"])
    return out


def _too_similar(a: str, b: str, threshold: float = 0.72) -> bool:
    """判断两个事实值是不是同一件事的不同说法。"""
    if a == b:
        return True
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def upsert_preferences(conn: sqlite3.Connection, prefs: list[dict[str, Any]], evidence: str = "") -> int:
    added = 0
    known = _prefs_by_domain(conn)
    for item in prefs:
        statement = textutil.normalize(str(item.get("statement", "")))
        if not statement:
            continue
        domain = item.get("domain", "其他")
        if any(_too_similar(statement, old) for old in known.get(domain, ())):
            continue          # 同一领域下已有高度相似的说法，不重复入库
        conn.execute(
            """INSERT INTO preferences (domain, statement, polarity, confidence, evidence)
               VALUES (?,?,?,?,?)
               ON CONFLICT(domain, statement) DO UPDATE SET
                 confidence = MAX(confidence, excluded.confidence),
                 evidence = COALESCE(preferences.evidence,'') || ',' || excluded.evidence""",
            (item.get("domain", "其他"), statement, item.get("polarity", "neutral"),
             float(item.get("confidence", 0.7)), evidence),
        )
        added += 1
        known.setdefault(domain, set()).add(statement)
    conn.commit()
    return added


def _prefs_by_domain(conn: sqlite3.Connection) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in conn.execute("SELECT domain, statement FROM preferences WHERE status='active'"):
        out.setdefault(row["domain"], set()).add(row["statement"])
    return out


def _reindex_facts(conn: sqlite3.Connection) -> None:
    indexed = {r["fact_id"] for r in conn.execute("SELECT fact_id FROM facts_fts")}
    for row in conn.execute("SELECT id, predicate, object FROM facts WHERE status='active'"):
        if row["id"] in indexed:
            continue
        tokens = textutil.tokenize(f"{row['predicate']} {row['object']}")
        conn.execute("INSERT INTO facts_fts (tokens, fact_id) VALUES (?,?)", (tokens, row["id"]))
    conn.commit()


# ---------------- 人设卡 ----------------

PERSONA_ORDER = (
    "姓名", "昵称", "身份", "年级专业", "籍贯", "城市", "学校", "公司", "职业",
    "到岗时间", "感情", "人际关系", "风格", "兴趣", "称呼习惯", "工作沟通", "雷区", "其他",
)


def set_persona(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """INSERT INTO persona (key, value, updated_at) VALUES (?,?,CURRENT_TIMESTAMP)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
        (key.strip(), value.strip()),
    )
    conn.commit()


def get_persona(conn: sqlite3.Connection) -> dict[str, str]:
    data = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM persona") if r["value"]}
    ordered = {k: data[k] for k in PERSONA_ORDER if k in data}
    ordered.update({k: v for k, v in data.items() if k not in PERSONA_ORDER})
    return ordered


def persona_block(persona: dict[str, str]) -> str:
    if not persona:
        return ""
    lines = "\n".join(f"- {k}：{v}" for k, v in persona.items())
    return f"【我是谁——回复必须符合这个身份】\n{lines}"


_CONTACT_SCENES: dict[str, str] | None = None
_GROUP_SCENES: dict[str, str] | None = None
_RELATIONS: dict[str, dict[str, str]] | None = None
_STYLE_PROFILE: dict | None = None

# 说话方式三档：跟谁说话决定用哪一档，档位内容是固定的，不随话题检索
TIER_CLOSE = "熟人"
TIER_PEER = "同龄不熟"
TIER_SENIOR = "年长或办正事"

CLOSE_SCENES = {"亲密关系", "高中同学", "初中同学", "广州宿舍", "清远宿舍", "家人", "朋友", "游戏群"}
SENIOR_SCENES = {"工作", "老师", "长辈", "商家", "驾校", "租房", "外卖", "校园墙", "广告", "系统号"}


def _load_json_map(filename: str) -> dict:
    path = ROOT / "data" / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _style_profile() -> dict:
    """本人说话方式档（data/style_profile.json）。"""
    global _STYLE_PROFILE
    if _STYLE_PROFILE is None:
        _STYLE_PROFILE = _load_json_map("style_profile.json") or {}
    return _STYLE_PROFILE


def tier_for_scene(scene: str) -> str:
    """关系大类折成三档说话方式。认不出大类的按同龄不熟处理。"""
    if scene in CLOSE_SCENES:
        return TIER_CLOSE
    if scene in SENIOR_SCENES:
        return TIER_SENIOR
    return TIER_PEER


def tier_samples(tier: str) -> list[str]:
    info = (_style_profile().get("档位") or {}).get(tier) or {}
    return [str(x) for x in (info.get("样例") or [])]


def style_block(tier: str) -> str:
    """拼这一轮要注入的说话方式档：总规律 + 这个场合的规矩 + 真实原话样例。"""
    prof = _style_profile()
    if not prof or not tier:
        return ""
    parts: list[str] = []
    base = prof.get("基础") or []
    if base:
        parts.append("【我平时怎么说话——按这个来，别写成客服】\n"
                     + "\n".join(f"- {x}" for x in base))
    info = (prof.get("档位") or {}).get(tier) or {}
    if info:
        lines = [f"【这一轮是什么场合：{info.get('何时用') or tier}】"]
        if info.get("称呼"):
            lines.append(f"- 怎么称呼他：{info['称呼']}")
        if info.get("语气"):
            lines.append(f"- 怎么说话：{info['语气']}")
        samples = info.get("样例") or []
        if samples:
            lines.append("- 我以前这么说过（学语感，别照抄内容）：\n"
                         + "\n".join(f"  · {s}" for s in samples))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _contact_scenes() -> dict[str, str]:
    """桌面分类表导出的 微信备注 → 大类。认人后优先用这个，不靠灌库时的会话名规则。"""
    global _CONTACT_SCENES
    if _CONTACT_SCENES is None:
        _CONTACT_SCENES = _load_json_map("contact_scenes.json")
    return _CONTACT_SCENES


def _group_scenes() -> dict[str, str]:
    global _GROUP_SCENES
    if _GROUP_SCENES is None:
        _GROUP_SCENES = _load_json_map("group_scenes.json")
    return _GROUP_SCENES


def _relations() -> dict[str, dict[str, str]]:
    """近圈薄卡：这个人是谁、怎么说、还在不在联系。"""
    global _RELATIONS
    if _RELATIONS is None:
        raw = _load_json_map("relations.json")
        _RELATIONS = {str(k): dict(v) for k, v in raw.items()} if raw else {}
    return _RELATIONS


_PAREN_NAME = re.compile(r"[（(【\[]([^)）】\]]+)[)）】\]]")
_DATEISH_NAME = re.compile(r"^[\d.]+$")
# 自报家门：「我是谢总」「我叫许伟」「这边是导师」
_SELF_INTRO = re.compile(r"(?:我是|我系|我叫|这边是|这里是|这是|我就是)\s*([^\s，。！？,.!?、；;]{1,16})")


def _name_aliases(name: str) -> list[str]:
    """谢总（12.22）→ 谢总；（嘎子）→ 嘎子。日期括号丢掉。"""
    n = textutil.normalize(name or "")
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = textutil.normalize(token).strip(" ·．.。~～-—_☆★♪✦💭☁")
        if token and token not in seen:
            seen.add(token)
            out.append(token)

    add(n)
    stripped = _PAREN_NAME.sub("", n).strip()
    add(stripped)
    cjk = "".join(ch for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    add(cjk)
    for inner in _PAREN_NAME.findall(n):
        inner = inner.strip()
        if _DATEISH_NAME.fullmatch(inner):
            continue
        add(inner)
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", inner):
            add(chunk)
    return out


def relation_for(contact: str) -> dict[str, str] | None:
    name = textutil.normalize(contact or "").strip()
    if not name:
        return None
    cards = _relations()
    if name in cards:
        return cards[name]
    hits: list[tuple[int, dict[str, str]]] = []
    for key, card in cards.items():
        aliases = _name_aliases(key)
        if name in aliases or textutil.normalize(key) == name:
            hits.append((len(key), card))
        elif any(name in a or a in name for a in aliases if len(a) >= 2):
            hits.append((len(key) + 1000, card))
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def relation_prompt_block(contact: str, scene: str = "", *, brief: bool = False) -> str:
    """给提示词用的「这个人是谁」。没薄卡就只落大类。

    brief=True 时不塞 who 正文——那是备忘，留给 who_is 工具，避免模型照抄。
    """
    name = (contact or "").strip()
    if not name:
        return ""
    card = relation_for(name) or {}
    scene = scene or card.get("scene") or _contact_scenes().get(name) or _group_scenes().get(name) or ""
    lines = [f"- 对方：{name}"]
    if scene:
        lines.append(f"- 大类：{scene}")
    if not brief and card.get("who"):
        lines.append(f"- 这个人：{card['who']}")
    if card.get("status") == "停":
        lines.append("- 近况：这条已经稀了或停了，不要装昨天还在聊。")
    elif card.get("status") == "活":
        lines.append("- 近况：还在联系。")
    how = card.get("how") or (tone_rule(scene) if scene else "")
    if how:
        lines.append(f"- 在这个人面前：{how}")
    return "【这个人是谁】\n" + "\n".join(lines)


def resolve_contact(conn: sqlite3.Connection, raw: str) -> str | None:
    """把「谢总」「小胖」对上会话名。对不上返回 None。"""
    name = textutil.normalize(raw or "").strip()
    if not name:
        return None
    row = conn.execute("SELECT name FROM conversations WHERE name = ?", (raw,)).fetchone()
    if row:
        return row["name"]
    row = conn.execute("SELECT name FROM conversations WHERE name = ?", (name,)).fetchone()
    if row:
        return row["name"]
    cards = _relations()
    for key in cards:
        aliases = _name_aliases(key)
        if name in aliases or textutil.normalize(key) == name:
            conv = conn.execute("SELECT name FROM conversations WHERE name = ?", (key,)).fetchone()
            return conv["name"] if conv else key
    exact: list[str] = []
    partial: list[tuple[int, str]] = []
    for row in conn.execute("SELECT name FROM conversations"):
        aliases = _name_aliases(row["name"])
        if name in aliases:
            exact.append(row["name"])
        elif any(len(a) >= 2 and (name in a or a in name) for a in aliases):
            partial.append((len(row["name"]), row["name"]))
    if exact:
        exact.sort(key=len)
        return exact[0]
    if partial:
        partial.sort()
        return partial[0][1]
    return None


def _clip(text: str, n: int = 60) -> str:
    t = textutil.redact(textutil.normalize(text or ""))
    return t if len(t) <= n else t[: n - 1] + "…"


def _people_path(name: str):
    safe = re.sub(r'[<>:"/\\|?*]', "_", (name or "").strip())
    return ROOT / "data" / "people" / f"{safe}.md"


_PLACEHOLDER_EXACT = {
    "[表情]", "[图片]", "[分享/文件/链接]", "[语音]", "[文本]", "[视频]",
    "[位置]", "[系统消息]", "[动画表情]", "[文件]", "[链接]", "[聊天记录]",
    "[视频号]", "[小程序]", "[音乐]", "[名片]",
}
_PLACEHOLDER_RE = re.compile(r"^\[[^\]]{1,16}\]$")


def _is_sample_noise(text: str) -> bool:
    t = textutil.normalize(text or "")
    if not t or len(t) <= 1:
        return True
    if t in _PLACEHOLDER_EXACT or _PLACEHOLDER_RE.fullmatch(t):
        return True
    return False


def _stats_line(stats: dict[str, Any]) -> str:
    n = int(stats.get("messages") or 0)
    if not n:
        return "记录很少"
    self_n = int(stats.get("self") or 0)
    other = int(stats.get("other") or max(0, n - self_n))
    first = stats.get("first") or ""
    last = stats.get("last") or ""
    weekly = stats.get("weekly") or 0
    return f"{n} 条（我 {self_n} / 对方 {other}）· {first} → {last} · 每周约 {weekly:.0f} 条"


def parse_people_md(text: str) -> dict[str, str]:
    """拆 data/people/*.md：front-matter + 正文。无第三方 YAML。"""
    meta: dict[str, str] = {}
    body = (text or "").strip()
    if body.startswith("---"):
        rest = body[3:]
        end = rest.find("\n---")
        if end != -1:
            for raw in rest[:end].splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            body = rest[end + 4:].strip()
    understanding = body
    if "## 依据" in body:
        understanding = body.split("## 依据", 1)[0].strip()
    if "## 历史" in understanding:
        understanding = understanding.split("## 历史", 1)[0].strip()
    meta["body"] = body
    meta["understanding"] = understanding
    return meta


def read_people_doc(name: str) -> dict[str, str]:
    """按联系人名读固化文件。对不上就按外号扫一遍。"""
    if not name:
        return {}
    path = _people_path(name)
    if path.exists():
        return parse_people_md(path.read_text(encoding="utf-8"))
    card = relation_for(name)
    if not card:
        return {}
    aliases = _name_aliases(name)
    for key in _relations():
        if key == name:
            continue
        if any(a in _name_aliases(key) for a in aliases if len(a) >= 2):
            p = _people_path(key)
            if p.exists():
                doc = parse_people_md(p.read_text(encoding="utf-8"))
                doc.setdefault("name", key)
                return doc
    return {}


def _text_weight(text: str) -> int:
    t = textutil.normalize(text or "")
    n = len(t)
    if n <= 2:
        return 1
    return n * n


def _best_run(rows: list, size: int = 10) -> list:
    """一段里取信息量最高的 size 条连续消息，避免整窗都是「噢噢」「来不来」。"""
    if len(rows) <= size:
        return list(rows)
    best_i, best_s = 0, -1
    for i in range(0, len(rows) - size + 1):
        score = sum(_text_weight(r["text"] or "") for r in rows[i:i + size])
        if score > best_s:
            best_s, best_i = score, i
    return list(rows[best_i:best_i + size])


def sample_person(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    """按 实施方案 6.1 取固化素材：7 窗 × 10 条连续 + 最近 26 条。"""
    resolved = resolve_contact(conn, name) or (name or "").strip()
    card = relation_for(resolved) or {}
    scene = scene_for_contact(conn, resolved) or card.get("scene") or ""
    stats = interaction_stats(conn, resolved)
    line = _stats_line(stats)
    empty = {
        "found": bool(resolved),
        "name": resolved,
        "scene": scene or "",
        "stats": line,
        "lines": [],
        "usable": 0,
        "source_hash": hashlib.sha1(b"").hexdigest()[:8],
        "too_few": True,
    }
    if not resolved:
        empty["found"] = False
        return empty
    row = conn.execute("SELECT id FROM conversations WHERE name=?", (resolved,)).fetchone()
    if not row:
        return empty
    rows = conn.execute(
        """SELECT id, role, text, ts, ts_epoch FROM episodes
           WHERE conv_id=? ORDER BY ts_epoch, id""",
        (row["id"],),
    ).fetchall()
    usable = [r for r in rows if not _is_sample_noise(r["text"] or "")]
    n = len(usable)
    if n < 40:
        blob = "\n".join(_format_sample_line(r) for r in usable)
        empty["usable"] = n
        empty["source_hash"] = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
        empty["too_few"] = True
        return empty
    head = usable[:-26] if n > 26 else usable
    hn = len(head)
    picked: list[Any] = []
    seen: set[int] = set()
    if hn:
        buckets = 7
        for i in range(buckets):
            a = int(i * hn / buckets)
            b = int((i + 1) * hn / buckets)
            chunk = head[a:b]
            if not chunk:
                continue
            for item in _best_run(chunk, 10):
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                picked.append(item)
    for item in usable[-26:]:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        picked.append(item)
    picked.sort(key=lambda r: (int(r["ts_epoch"] or 0), int(r["id"])))
    lines = [_format_sample_line(item) for item in picked]
    blob = "\n".join(lines)
    return {
        "found": True,
        "name": resolved,
        "scene": scene or "",
        "stats": line,
        "lines": lines,
        "usable": n,
        "source_hash": hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8],
        "too_few": False,
    }


def _format_sample_line(row: sqlite3.Row | dict[str, Any]) -> str:
    who = "我" if row["role"] == "self" else "对方"
    text = _clip(row["text"] or "", 70)
    ts = int(row["ts_epoch"] or 0)
    month = time.strftime("%Y-%m", time.localtime(ts)) if ts else ""
    if not month and row["ts"]:
        month = str(row["ts"])[:7]
    return f"{month} {who}：{text}" if month else f"{who}：{text}"


def people_note(name: str) -> str:
    """固化后的「我眼里的他」（只正文，不含 front-matter）。"""
    return (read_people_doc(name).get("understanding") or "").strip()


def person_index() -> str:
    """近圈一人一行。优先读固化 oneliner，没有文件就用薄卡第一句。"""
    cards = _relations()
    if not cards:
        return ""
    lines = ["【人物索引（需要细节时用 who_is / recent_with 去查，不要把这当稿子念）】"]
    for name, card in cards.items():
        doc = read_people_doc(name)
        oneliner = (doc.get("oneliner") or "").strip()
        if not oneliner:
            who = (card.get("who") or "").split("。")[0].split("；")[0].strip()
            oneliner = _clip(who, 40)
        display = name
        lines.append(f"- {display}：{oneliner}" if oneliner else f"- {display}")
    return "\n".join(lines)


def interaction_stats(conn: sqlite3.Connection, contact: str) -> dict[str, Any]:
    """跟这个人互动的数字事实：跨度、条数、是否还在聊。不列群名。"""
    name = resolve_contact(conn, contact) or (contact or "").strip()
    if not name:
        return {}
    row = conn.execute("SELECT id FROM conversations WHERE name = ?", (name,)).fetchone()
    if not row:
        return {"name": name, "messages": 0}
    cid = row["id"]
    agg = conn.execute(
        """SELECT COUNT(*) AS n,
                  SUM(CASE WHEN role='self' THEN 1 ELSE 0 END) AS self_n,
                  MIN(ts_epoch) AS first_ts,
                  MAX(ts_epoch) AS last_ts
           FROM episodes WHERE conv_id = ?""",
        (cid,),
    ).fetchone()
    n = int(agg["n"] or 0)
    self_n = int(agg["self_n"] or 0)
    first_ts = int(agg["first_ts"] or 0)
    last_ts = int(agg["last_ts"] or 0)
    now = int(time.time())
    d90 = 0
    if last_ts:
        d90 = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE conv_id=? AND ts_epoch > ?",
            (cid, now - 90 * 86400),
        ).fetchone()[0]
    days = max(1.0, (last_ts - first_ts) / 86400) if last_ts and first_ts else 1.0
    weekly = n / (days / 7) if days > 7 else float(n)
    first = time.strftime("%Y-%m", time.localtime(first_ts)) if first_ts else ""
    last = time.strftime("%Y-%m-%d", time.localtime(last_ts)) if last_ts else ""
    ago_days = int((now - last_ts) / 86400) if last_ts else None
    note = ""
    if n:
        note = (f"从 {first} 聊到 {last}，共 {n:,} 条（我 {self_n:,} / 对方 {n - self_n:,}），"
                f"平均每周约 {weekly:.0f} 条，最近 90 天 {d90:,} 条")
        if ago_days is not None:
            if ago_days <= 0:
                note += "，今天还在聊"
            elif ago_days < 30:
                note += f"，最后一次大约 {ago_days} 天前"
            else:
                note += f"，最后一次大约 {ago_days // 30} 个月前"
    return {
        "name": name, "messages": n, "self": self_n, "other": n - self_n,
        "first": first, "last": last, "weekly": round(weekly, 1),
        "last_90d": d90, "days_since": ago_days, "note": note,
    }


def who_is(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    """优先读 data/people/*.md；没有文件再回退薄卡 + 互动统计。"""
    raw = (name or "").strip()
    resolved = resolve_contact(conn, raw) or raw
    if not resolved:
        return {"found": False, "name": raw, "reason": "没有这个人的记录"}
    scene = scene_for_contact(conn, resolved)
    card = relation_for(resolved) or {}
    stats = interaction_stats(conn, resolved)
    doc = read_people_doc(resolved)
    found = bool(doc or card or scene or (stats.get("messages") or 0))
    if not found:
        return {"found": False, "name": resolved, "reason": "通讯录里没有这个人"}
    oneliner = (doc.get("oneliner") or "").strip()
    if not oneliner and card.get("who"):
        oneliner = _clip(str(card["who"]).split("。")[0], 40)
    understanding = (doc.get("understanding") or "").strip()
    if len(understanding) > 400:
        understanding = understanding[:399] + "…"
    payload = {
        "found": True,
        "name": resolved,
        "scene": (doc.get("scene") or scene or ""),
        "oneliner": oneliner,
        "understanding": understanding,
        "stats": doc.get("stats") or _stats_line(stats),
        "hint": "这是备忘，回答时用自己的话讲，不要照抄，也不要念群名。括号里的数字是备注不是生日。",
    }
    if not understanding and card:
        payload["card"] = {k: card[k] for k in ("who", "how", "status") if card.get(k)}
    return payload


def recent_with(conn: sqlite3.Connection, name: str, limit: int = 8) -> dict[str, Any]:
    """我跟这个人最近几轮聊天。每条截断 60 字，总量不超过 600 字。"""
    resolved = resolve_contact(conn, name) or (name or "").strip()
    try:
        limit = max(1, min(12, int(limit)))
    except (TypeError, ValueError):
        limit = 8
    if not resolved:
        return {"found": False, "name": name, "turns": [], "reason": "没有这个人"}
    row = conn.execute("SELECT id FROM conversations WHERE name = ?", (resolved,)).fetchone()
    if not row:
        return {"found": False, "name": resolved, "turns": [], "reason": "没有这个人的会话"}
    rows = conn.execute(
        """SELECT role, text, ts FROM episodes
           WHERE conv_id=? ORDER BY ts_epoch DESC, id DESC LIMIT ?""",
        (row["id"], limit),
    ).fetchall()
    turns: list[dict[str, str]] = []
    total = 0
    for item in reversed(rows):
        text = _clip(item["text"] or "", 60)
        if total + len(text) > 600:
            text = text[: max(0, 600 - total - 1)] + "…"
        total += len(text)
        turns.append({
            "who": "我" if item["role"] == "self" else "他",
            "text": text,
            "time": item["ts"] or "",
        })
        if total >= 600:
            break
    if not turns:
        return {"found": False, "name": resolved, "turns": [], "reason": "没有聊天记录"}
    return {"found": True, "name": resolved, "turns": turns}


def search_chats(
    conn: sqlite3.Connection,
    keyword: str,
    contact: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    """在聊天记录里搜关键词。可限定某个人。"""
    try:
        limit = max(1, min(12, int(limit)))
    except (TypeError, ValueError):
        limit = 8
    query = _fts_query(keyword or "")
    if not query:
        return {"found": False, "hits": [], "reason": "没有可搜的词"}
    resolved = resolve_contact(conn, contact) if contact else ""
    sql = """SELECT e.text, e.role, e.ts, c.name AS name
             FROM episodes_fts ft JOIN episodes e ON e.id = ft.episode_id
             JOIN conversations c ON c.id = e.conv_id
             WHERE episodes_fts MATCH ?"""
    params: list[Any] = [query]
    if resolved:
        sql += " AND c.name = ?"
        params.append(resolved)
    sql += " ORDER BY bm25(episodes_fts) LIMIT ?"
    params.append(limit)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return {"found": False, "hits": [], "reason": "没有记录"}
    hits: list[dict[str, str]] = []
    total = 0
    for row in rows:
        text = _clip(row["text"] or "", 60)
        if total + len(text) > 600:
            break
        total += len(text)
        hits.append({
            "contact": row["name"],
            "who": "我" if row["role"] == "self" else "他",
            "text": text,
            "time": row["ts"] or "",
        })
    if not hits:
        return {
            "found": False, "hits": [], "reason": "没有搜到",
            "keyword": keyword, "contact": resolved or contact,
        }
    return {"found": True, "hits": hits, "keyword": keyword,
            "contact": resolved or contact}


def my_facts(conn: sqlite3.Connection, topic: str = "") -> dict[str, Any]:
    """关于我的事实和偏好。有主题就按主题滤，没有就给人设卡摘要。"""
    topic = (topic or "").strip()
    persona = get_persona(conn)
    facts: list[str] = []
    prefs: list[str] = []
    if topic:
        query = _fts_query(topic)
        if query:
            try:
                for row in conn.execute(
                    """SELECT f.predicate, f.object FROM facts_fts ft
                       JOIN facts f ON f.id = ft.fact_id
                       WHERE facts_fts MATCH ? AND f.status='active'
                       ORDER BY rank LIMIT 8""",
                    (query,),
                ):
                    gate = GATED_PREDICATES.get(row["predicate"])
                    if gate and not any(word in topic for word in gate):
                        continue
                    facts.append(f"{row['predicate']}：{_clip(row['object'], 40)}")
            except sqlite3.OperationalError:
                pass
        for row in conn.execute(
            """SELECT statement FROM preferences WHERE status='active'
               ORDER BY id DESC LIMIT 80"""
        ):
            if any(t in row["statement"] for t in textutil.tokenize(topic).split() if len(t) >= 2):
                prefs.append(_clip(row["statement"], 40))
            if len(prefs) >= 6:
                break
    else:
        for key in ("身份", "年级专业", "学校", "公司", "职业", "感情", "兴趣"):
            if persona.get(key):
                facts.append(f"{key}：{_clip(persona[key], 40)}")
        for row in conn.execute(
            """SELECT statement FROM preferences WHERE status='active'
               ORDER BY confidence DESC, id DESC LIMIT 6"""
        ):
            prefs.append(_clip(row["statement"], 40))
    return {
        "found": bool(facts or prefs or persona),
        "facts": facts[:8],
        "preferences": prefs[:6],
        "hint": "用自己的话讲，不要列清单。",
        **({} if (facts or prefs or persona) else {"reason": "没有相关事实"}),
    }


def scene_for_contact(conn: sqlite3.Connection, contact: str) -> str:
    """按联系人名字查出他所属的场景，用来切换口吻。"""
    if not contact:
        return ""
    overlay = _contact_scenes()
    if contact in overlay:
        return overlay[contact]
    groups = _group_scenes()
    if contact in groups:
        return groups[contact]
    row = conn.execute(
        """SELECT e.scene, COUNT(*) AS n FROM episodes e JOIN conversations c ON c.id = e.conv_id
           WHERE c.name = ? AND e.scene <> '' GROUP BY e.scene ORDER BY n DESC LIMIT 1""",
        (contact,),
    ).fetchone()
    if row:
        return row["scene"]
    row = conn.execute(
        """SELECT e.scene, COUNT(*) AS n FROM episodes e JOIN conversations c ON c.id = e.conv_id
           WHERE c.name LIKE ? AND e.scene <> '' GROUP BY e.scene ORDER BY n DESC LIMIT 1""",
        (f"%{contact}%",),
    ).fetchone()
    return row["scene"] if row else ""


def scenes_of(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """按场景统计条数，网页右边那个分布图用。"""
    return [(r["scene"], r["n"]) for r in conn.execute(
        """SELECT scene, COUNT(*) AS n FROM episodes WHERE IFNULL(scene,'') <> ''
           GROUP BY scene ORDER BY n DESC""")]


def hit_info(conn: sqlite3.Connection, contact: str, scene: str = "") -> dict[str, str]:
    """这轮认到的是哪一层：内层薄卡 / 外层大类 / 没命中。

    只用来告诉界面「命中的是哪一层」，不是「答案的依据」。
    """
    name = (contact or "").strip()
    if not name:
        return {"layer": "self", "label": ""}
    if relation_for(name):
        return {"layer": "card", "label": name}
    label = (scene or scene_for_contact(conn, name) or "").strip()
    if label:
        return {"layer": "scene", "label": label}
    return {"layer": "none", "label": ""}


# ---------------- 检索 ----------------

def _fts_query(text: str, max_terms: int = 24) -> str:
    terms = textutil.tokenize(text).split()[:max_terms]
    return " OR ".join(f'"{t}"' for t in terms) if terms else ""


def _normalize_ranks(ranks: list[float]) -> list[float]:
    """把一批 bm25 分数（越负越相关）映射到 0-1。

    用同一批候选做 min-max 归一化，而不是逐条做 1/(1+|x|)：
    后者在候选很多时会全部挤在 0.7 附近，权重再混合就失去了区分度。
    """
    if not ranks:
        return []
    best, worst = min(ranks), max(ranks)
    if worst - best < 1e-9:
        return [1.0] * len(ranks)
    return [(worst - r) / (worst - best) for r in ranks]


def retrieve_facts(
    conn: sqlite3.Connection,
    situation: str,
    *,
    contact: str = "",
    k: int = 8,
) -> list[dict[str, Any]]:
    """这轮要带上的记忆：跟这个人的关系卡 + 关键词命中的事实。

    说话方式不走这里——那是一份固定的档（data/style_profile.json），
    不按关键词去历史里捞原话。
    """
    return _retrieve_facts(
        conn, _fts_query(situation), k, contact=contact, situation=situation)


def _pinned_core_facts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """兜底身份锚点：只在人设卡还是空的库里用。

    人设卡填好之后就不再需要——感情、年级、专业、籍贯、单位这些内容人设卡里都有，
    而人设卡每一轮本来就会注入提示词，再塞一份只是重复。
    """
    out: list[dict[str, Any]] = []
    for pred in ("感情", "年级", "专业", "籍贯", "实习单位"):
        row = conn.execute(
            """SELECT id, predicate, object, confidence FROM facts
               WHERE predicate=? AND status='active' ORDER BY confidence DESC LIMIT 1""",
            (pred,),
        ).fetchone()
        if row:
            out.append({
                "id": row["id"], "predicate": row["predicate"],
                "object": textutil.redact(row["object"]),
                "confidence": row["confidence"] or 0.9, "score": 1.0, "pinned": True,
            })
    return out


def _persona_filled(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) AS n FROM persona WHERE value <> ''").fetchone()
    return bool(row and row["n"])


def _relation_facts(conn: sqlite3.Connection, contact: str) -> list[dict[str, Any]]:
    card = relation_for(contact)
    if card and card.get("who"):
        label = (contact or "").strip()
        return [{
            "id": 0, "predicate": "关系", "object": f"{label}：{card['who']}",
            "confidence": 0.99, "score": 1.0, "pinned": True,
        }]
    if not contact:
        return []
    key = contact[:8]
    rows = []
    for row in conn.execute(
        """SELECT id, predicate, object, confidence FROM facts
           WHERE predicate='关系' AND status='active' AND object LIKE ?
           ORDER BY confidence DESC LIMIT 3""",
        (f"%{key}%",),
    ):
        rows.append({
            "id": row["id"], "predicate": row["predicate"],
            "object": textutil.redact(row["object"]),
            "confidence": row["confidence"] or 0.8, "score": 0.9, "pinned": True,
        })
    return rows


def _mentioned_relation_facts(text: str) -> list[dict[str, Any]]:
    """早安问「谢总是谁」时，把近圈薄卡钉进事实，不靠 FTS 碰运气。"""
    blob = textutil.normalize(text or "")
    if not blob:
        return []
    # 对方自报名字时只认这个名字：有人自称「天线宝宝」，不该把薄卡「宝宝」捞出来
    claimed = ""
    intro = _SELF_INTRO.search(blob)
    if intro:
        claimed = intro.group(1).strip()
    hits: list[tuple[int, str, dict[str, str]]] = []
    for name, card in _relations().items():
        who = (card.get("who") or "").strip()
        if not who:
            continue
        aliases = [
            a for a in _name_aliases(name)
            if len(a) >= 2 or (len(a) == 1 and "\u4e00" <= a <= "\u9fff")
        ]
        if claimed:
            matched = [a for a in aliases if a == claimed or claimed.startswith(a)]
        else:
            matched = [a for a in aliases if a in blob]
        if not matched:
            continue
        hits.append((max(len(a) for a in matched), name, card))
    hits.sort(reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, name, card in hits:
        if name in seen:
            continue
        if any(name != other and name in other for _, other, _ in hits):
            continue
        seen.add(name)
        out.append({
            "id": 0, "predicate": "关系", "object": f"{name}：{card['who']}",
            "confidence": 0.99, "score": 1.0, "pinned": True,
        })
        if len(out) >= 5:
            break
    return out


def _retrieve_facts(
    conn: sqlite3.Connection, query: str, k_facts: int, *, contact: str = "",
    situation: str = "",
) -> list[dict[str, Any]]:
    raw_pinned = _relation_facts(conn, contact) + _mentioned_relation_facts(situation)
    if not _persona_filled(conn):
        raw_pinned = _pinned_core_facts(conn) + raw_pinned
    pinned: list[dict[str, Any]] = []
    seen_pin: set[str] = set()
    for item in raw_pinned:  # 同一张卡可能被两条路都捞到，去个重
        if item["object"] in seen_pin:
            continue
        seen_pin.add(item["object"])
        pinned.append(item)
    seen_obj = {f["object"] for f in pinned}
    if not query:
        return pinned
    sql = """SELECT f.id, f.predicate, f.object, f.confidence, bm25(facts_fts) AS rank
             FROM facts_fts ft JOIN facts f ON f.id = ft.fact_id
             WHERE facts_fts MATCH ? AND f.status='active'
             ORDER BY rank LIMIT ?"""
    facts: list[dict[str, Any]] = []
    for row in conn.execute(sql, (query, k_facts * 4)):
        if row["predicate"] == "关系":
            # 关系只按名字注入（上面那两条路），不许关键词碰运气——
            # 否则别人说「心情不好」都会把谢总的卡捞出来
            continue
        gate = GATED_PREDICATES.get(row["predicate"])
        if gate and not any(word in situation for word in gate):
            continue
        obj = textutil.redact(row["object"])
        if obj in seen_obj:
            continue
        facts.append({
            "id": row["id"], "predicate": row["predicate"], "object": obj,
            "confidence": row["confidence"] or 0.5, "rank": row["rank"],
        })
        seen_obj.add(obj)
    for item, similarity in zip(facts, _normalize_ranks([i["rank"] for i in facts])):
        item["score"] = round(RETRIEVAL_WEIGHTS["similarity"] * similarity
                              + RETRIEVAL_WEIGHTS["quality"] * item["confidence"], 4)
        item.pop("rank", None)
    merged = pinned + facts
    return merged[: max(k_facts, len(pinned))]


# ---------------- 提示组装 ----------------

# 语气档位：按关系场景给不同的表达约束。
# 这是本项目的核心设计之一——语气由亲疏度决定，而不是把所有人当成同一个人。
TONE_PROFILES: dict[str, str] = {
    "亲密关系": "历史口吻极简、黏，短句，会认错。当前单身已分手，不要说自己现在有对象；对方用这个身份找过来仍用这套口吻。",
    "家人": "随意、报平安式，别啰嗦；不客套。",
    "高中同学": "熟人模式：短句、直接，可以互怼和开玩笑；该拒绝就拒绝，不用铺垫。",
    "初中同学": "熟人模式：简短、随意。昊、孜然是很要好的女性朋友，不是普通同学，也不要当成对象。",
    "大学同学": "朋友模式：随意，可以吐槽课业和实习；比高中同学正常一点，但同样不客套。",
    "广州宿舍": "舍友模式：比普通大学同学更熟、更碎、更直接；作息、出门、带饭这类事不用说完整。",
    "清远宿舍": "舍友模式：碎、直接，作息出门带饭不用说完整；清远校区宿舍，别当成广州比奇堡那一拨。",
    "同学": "半熟模式：语气平和，保持同学间的分寸，不热络也不生硬。",
    "朋友": "熟人模式：直接、轻松，允许用网络梗。",
    "工作": "客气且有分寸：称呼用「您」，措辞收敛，给明确答复或时间点；"
            "但不写「非常感谢您的支持」这类模板句。",
    "长辈": "客气、尊重、简短；不卖梗，不用网络用语，称呼要恰当。",
    "初识": "客气但自然：有基本礼貌，不热络、不追问隐私、不主动开玩笑。",
    "游戏群": "游戏群口吻：短、快、梗多，可以不完整句。",
    "营销群": "对商家的口吻：直接问价格或说明来意，不寒暄。",
    "自留地": "这是本人给自己记笔记的场景，不属于对话。",
    "部门": "半熟同学：有组织里的分寸，不跟高中同学那样互怼，也不客套成工作。",
    "老师": "客气、尊重、简短；称呼恰当，不卖梗。",
    "商家": "直接说要干什么或问价格，不寒暄。",
    "广告": "当广告处理：能回就短回，不想理就敷衍，不热络。",
    "校园墙": "跟校园号办事，短，把事情说清。",
    "羽毛球": "球友：约球、场地，短，不必很熟。",
    "游戏": "一起打过游戏，可以短、可以用梗，但别当成很熟的同学。",
    "外卖": "因外卖认识，短，把事情说完就行。",
    "三下乡": "一起出过队，同学分寸，不必很黏。",
    "驾校": "办事：报名、练车，短，把事情说清。",
    "租房": "办事，短，把事情说清。",
    "不熟": "知道是通过谁认识的，但并不熟。客气、短、不热络、不开玩笑、不装同学。",
    "忘记了": (
        "微信里有这个人，但现实中想不起来是谁。"
        "只说：不好意思，我忘记你是谁了，告诉我你是谁，我们可以再认识一次。"
        "不要根据历史聊天装作还认识，不要回答他提的事情。"
    ),
    "骗子": "警惕，不要给个人信息，不要继续聊正事。",
    "系统号": "系统通知，不属于私人对话。",
    "自己": "自己的号或笔记，不属于对外对话。",
    "活动": "一起出过门或办过活动，随意、短，把时间地点说清就行，不必很黏。",
    "无用": "这个群没有借鉴意义。不要学里面的口吻，按不熟处理，能不问就少说。",
}


def tone_rule(scene: str) -> str:
    if not scene:
        return "按给出的历史原话判断语气，不要比历史更正式，也不要更热情。"
    return TONE_PROFILES.get(scene, TONE_PROFILES["同学"])


