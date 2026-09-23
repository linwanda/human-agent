"""晚风 Agent 内核：两个角色，一套记忆。

晚风（persona）：就是林万达本人。谁跟它聊，都相当于跟本人说话；
    对方是谁决定了它用什么语气。直接回一句，连续对话。

早安（assistant）：他的私人助理，不是替身。掌握真实情况，
    替他办事、给建议、做安排，不冒充本人。

两个角色共用同一套记忆（人设卡 + 事实 + 偏好 + 关系 + 风格样例），
区别在于身份定位和输出要求。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any

from . import kb, textutil
from .llm import LLM, LLMError, Reply, StubLLM, ToolCall

PERSONA_NAME = "晚风"
ASSISTANT_NAME = "早安"
AGENT_NAME = PERSONA_NAME  # 兼容旧调用：默认指晚风
BRAND = "晚风 Agent"

MODE_PERSONA = "persona"
MODE_ASSISTANT = "assistant"


def display_name(mode: str) -> str:
    return ASSISTANT_NAME if mode == MODE_ASSISTANT else PERSONA_NAME

# 自报家门的说法：我是谢总 / 我系谢总 / 我叫许伟 / 这边是导师 / 谢总啊
SELF_ID_PATTERNS = (
    re.compile(r"(?:我是|我系|我叫|这边是|这里是|这是|我就是)\s*([^\s，。！？,.!?、；;]{1,16})"),
    re.compile(r"^([^\s，。！？,.!?、；;]{1,16})(?:啊|呀|在此|在这|来了|报道)"),
)
SELF_WORDS = {"本人", "你自己", "你", "我", "林万达", "晚风", "早安"}
_PAREN = re.compile(r"[（(【\[]([^)）】\]]+)[)）】\]]")
_DATEISH = re.compile(r"^[\d.]+$")
_TAIL_PARTICLE = re.compile(r"[啊呀呢吧嘛哦噢哈]+$")
_ALIAS_DECOR = " ·．.。~～-—_☆★♪✦"

FORGOT_REPLY = "不好意思，我忘记你是谁了，告诉我你是谁，我们可以再认识一次。"


def _relation_notes_block(items: list[dict[str, Any]]) -> str:
    """关系卡不塞成一条事实，改成要点备忘，并说清是「拿来讲的」不是「拿来念的」。"""
    lines = ["【我关于这几个人的备忘（是我自己记的要点，不是现成的说法：讲的时候用你自己的话，"
             "别照抄，也别念成清单）】"]
    for item in items:
        obj = str(item.get("object") or "")
        name, sep, body = obj.partition("：")
        if not sep:
            name, sep, body = obj.partition(":")
        if not sep:
            name, body = "", obj
        # 括号里是日期 → 备注，丢掉；是外号 → 留着（谢总（12.22）→谢总，空想号MEK(小胖) 保留）
        name = name.strip()
        inner = [x.strip() for x in _PAREN.findall(name)]
        if inner and all(_DATEISH.match(x) for x in inner):
            name = _PAREN.sub("", name).strip(_ALIAS_DECOR) or name
        lines.append(f"* {name}" if name else "* 这个人")
        for frag in re.split(r"[。；;]", body):
            frag = frag.strip()
            if len(frag) >= 2:
                lines.append(f"  · {frag}")
    return "\n".join(lines)


def _relevant_preferences(conn: sqlite3.Connection, message: str, limit: int = 10) -> list[str]:
    """只带跟这轮话题有关的偏好。以前每轮硬塞 20 条，噪声比信息多。"""
    rows = [r["statement"] for r in conn.execute(
        "SELECT statement FROM preferences WHERE status='active' ORDER BY id DESC LIMIT 300")]
    terms = [t for t in textutil.tokenize(message or "").split() if len(t) >= 2]
    if not terms:
        return []
    hits: list[str] = []
    for statement in rows:
        if any(t in statement for t in terms):
            hits.append(statement)
            if len(hits) >= limit:
                break
    return hits

# 对方话里这些词说明他比你年长、或者在跟你办正事，说话要更规矩一档
SENIOR_HINTS = (
    "老师", "师兄", "师姐", "老板", "师傅", "教练", "主任", "校长", "经理",
    "人事", "hr", "导员", "教授", "您",
)


def stranger_tier(message: str) -> str:
    """通讯录里没有这个人时，从他自己的话里看该用哪一档说话。"""
    text = textutil.normalize(message or "").lower()
    if any(hint in text for hint in SENIOR_HINTS):
        return kb.TIER_SENIOR
    return kb.TIER_PEER


def _alias_token(text: str) -> str:
    return textutil.normalize(text or "").strip(_ALIAS_DECOR)


def contact_keys(name: str) -> list[str]:
    """微信备注里括号可能是日期，也可能是外号：～天影(嘎子) → 天影、嘎子。"""
    n = textutil.normalize(name or "")
    keys: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = _alias_token(token)
        cjk_one = len(token) == 1 and "\u4e00" <= token <= "\u9fff"
        if (len(token) >= 2 or cjk_one) and token not in seen:
            seen.add(token)
            keys.append(token)

    add(_PAREN.sub("", n))
    for inner in _PAREN.findall(n):
        inner = inner.strip()
        if _DATEISH.fullmatch(inner):
            continue
        add(inner)
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", inner):
            add(chunk)
    add(n)
    return keys


def _lookup_key(raw: str) -> str:
    key = _alias_token(raw)
    key = _TAIL_PARTICLE.sub("", key).strip(_ALIAS_DECOR)
    if len(key) >= 2:
        return key
    if len(key) == 1 and "\u4e00" <= key <= "\u9fff":
        return key
    return ""


def match_contact(conn: sqlite3.Connection, raw: str) -> str | None:
    """把一个称呼对上具体会话。外号、括号里的备注、去日期后的名字都能对。"""
    key = _lookup_key(raw)
    if not key:
        return None
    exact: list[str] = []
    partial: list[tuple[int, str]] = []
    for row in conn.execute("SELECT name FROM conversations"):
        name = row["name"]
        keys = contact_keys(name)
        if not keys:
            continue
        if key in keys or textutil.normalize(name) == textutil.normalize(raw):
            exact.append(name)
        elif any(key in item for item in keys):
            partial.append((min(len(item) for item in keys if key in item), name))
    if exact:
        exact.sort(key=len)
        return exact[0]
    if partial:
        partial.sort()
        return partial[0][1]
    return None


def infer_partner(conn: sqlite3.Connection, message: str) -> str:
    """从对方这句话里推断他是谁。推断不出来就返回空串。"""
    text = textutil.normalize(message)
    if not text:
        return ""
    for pattern in SELF_ID_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1).strip()
        if candidate in SELF_WORDS:
            return ""
        hit = match_contact(conn, candidate)
        if hit:
            return hit
    # 问完「你谁啊」之后，对方经常直接回一个名字或外号，没有「我是」
    if len(text) <= 16 and "\n" not in text:
        hit = match_contact(conn, text)
        if hit:
            return hit
    return ""

AGENT_SYSTEM = """你是「晚风」，就是林万达本人。

你不是助手，也不是「另一个我」这种说法。别人找你，等于找他本人。别人问你话，你用自己的方式回答：

1. 始终第一人称，像本人在微信里说话：短句、口语、不铺垫、不写小作文。
2. 直接回一句能发出去的话，不要给若干候选让对方挑。
3. 语气按「现在跟你说话的人」调整，规则在下面给出来了。
4. 认不出对方是谁的时候，不要追问「你谁啊」，也不要说「不认识」「我不记得你」：
   直接按「不熟的人」那一档回他问的事。对方报了名字，就用这个称呼接着聊，不要质疑。
5. 涉及他的事实（在哪、做什么、认识谁、什么态度），只能使用给出的事实。
   没有依据就直说不知道或记不清，不要编。
6. 近圈谁是谁以人设卡「人际关系」和「这个人是谁」为准；记录少的人不要借用熟人的历史口吻。
7. 不要自称 AI、模型、助手，也不要解释自己的回答方式。
8. 不写客套模板（「您好，感谢您的来信」这类），不堆敬语。
9. 遇到他也不知道的事，就像他本人那样直说，比如「这个我还真不知道」。
10. 别人问起某个人（这人是谁、跟我什么关系），先形成你自己的判断再用自己的话说：
   他是谁、跟我什么关系、在我这儿是什么位置。备忘不是稿子——别出现「关系：」这种
   字段名，也别把群名一串念出来。可以像人一样反问一句「你找他吗」，但不要追问对方是谁。
11. 你有工具。问某人是谁/什么关系时必须用 who_is，人物索引那一行不够当答案，不要靠它编。
   问最近聊了啥用 recent_with；问某话题聊过没用 search_chats；问我的情况或偏好（喜欢什么、
   平时怎样）必须用 my_facts，不要只靠人设卡里的兴趣一行。闲聊（在吗、嗯、哈哈、好）不要
   调工具。不要在回复里提到工具名或「我去查一下」。
12. 名字后面括号里的数字是备注，不是生日。"""

ASSISTANT_SYSTEM = """你是「早安」，林万达的私人助理。

你不是他本人，不要冒充晚风，也不要说自己就是林万达。你了解他的真实情况——学校、实习、作息、正在做的事、
偏好、人际关系——用这些信息替他办事、给建议、做安排。

要求：
1. 说话直接、简洁，不要客套，不要「作为一个 AI 助手」这类开场白。
2. 给建议要落到能执行：先做什么、什么时候做、大概多久，不要给空话。
3. 已经知道的信息直接用，不要再问他一遍。
4. 不确定的事直说不确定，不要编。
5. 需要他拍板的地方，给两三个选项并说明差别，不要只给一个方案。
6. 篇幅跟着事情走：简单的事一句话，复杂的事可以分步骤，但别写废话。
7. 别人找过来要跟「林万达本人」说话时，应让他们去找晚风，而不是你顶上去装本人。
8. 讲人和事之前先自己读懂：先判断他为什么问（想了解、想联系、想让你办事、还是随口一提），
   再开口。可以先给出你的理解，再补一两个细节；可以往下推一步（「你找他吗」「要我帮你拟条消息吗」）。
   备忘不是稿子——不要照抄，不要出现「关系：」「话题：」这类字段名，也不要把群名一串念出来。
9. 名字后面括号里的日期是备注，不是生日，别当生日说。
10. 他说过希望你语气活跃、幽默、直接，表情想用就用——按这个来，别端着。
11. 你有工具。问某人是谁/什么关系时必须用 who_is，人物索引那一行不够当答案，不要靠它编。
   问最近聊了啥用 recent_with；问某话题聊过没用 search_chats；问他的情况或偏好（喜欢什么、
   平时怎样）必须用 my_facts，不要只靠人设卡里的兴趣一行。闲聊不要调工具。不要在回复里提到工具名。
12. 要点不够就先查再讲，不要靠猜，也不要把人物索引里的一行扩写成一篇。"""

NO_MARKDOWN = """
【排版】这是聊天窗口，不是文档：不要用 markdown 语法（不要 **、##、- 、>）。
要点就写成「1. 2. 3.」，要强调就把话说清楚，不要加星号。"""

# 本人来问它的时候，用什么口吻
SELF_TONE = "直接回答，不用客套，也不用解释；像自言自语或者跟熟人说话那样。"


def partner_label(conn: sqlite3.Connection, partner: str) -> tuple[str, str, str]:
    """返回 (显示名, 场景, 语气规则)。partner 为空表示本人在跟它说话。"""
    partner = (partner or "").strip()
    if not partner or partner in {"本人", "我", PERSONA_NAME, ASSISTANT_NAME, "林万达"}:
        return "本人（林万达）", "本人", SELF_TONE
    scene = kb.scene_for_contact(conn, partner) or ""
    card = kb.relation_for(partner)
    tone = (card.get("how") if card else "") or kb.tone_rule(scene)
    return partner, scene or "未分类", tone


KEEP_HISTORY = 12  # 最近 6 轮（每人一条）原文
COMPACT_KEYS = (
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
)
COMPACT_SYSTEM = """你在压缩一段对话里更早的部分，给后续模型当备忘，不是给用户看的。
硬要求：
1. 先保留「旧摘要」里已有的信息，再用「新溢出的对话」补充或修正。
2. 只依据材料，不许编。材料没有的节写「无」。
3. 只输出 JSON，键必须正好是：Goal、Constraints & Preferences、Progress、Key Decisions、Next Steps、Critical Context。
4. 每节不超过 80 字，不要小标题以外的废话。"""


def _turns_blob(items: list[dict[str, str]], clip: int = 40) -> str:
    bits: list[str] = []
    for turn in items:
        role = "他" if turn.get("role") == "user" else "我"
        text = textutil.redact(str(turn.get("text") or "")).replace("\n", " ")
        if text:
            bits.append(f"{role}：{text[:clip]}")
    blob = "；".join(bits)
    return blob[:799] + "…" if len(blob) > 800 else blob


def _parse_summary(text: str) -> dict[str, str]:
    out = {key: "无" for key in COMPACT_KEYS}
    if not (text or "").strip():
        return out
    current = ""
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        hit = ""
        for key in COMPACT_KEYS:
            for prefix in (key + "：", key + ":"):
                if line.startswith(prefix):
                    hit = key
                    line = line[len(prefix):].strip()
                    break
            if hit:
                break
        if hit:
            if current:
                out[current] = "；".join(buf).strip() or "无"
            current, buf = hit, ([line] if line else [])
        elif current:
            buf.append(line)
    if current:
        out[current] = "；".join(buf).strip() or "无"
    return out


def _render_summary(sections: dict[str, str]) -> str:
    lines = []
    for key in COMPACT_KEYS:
        val = textutil.redact((sections.get(key) or "无").strip()) or "无"
        if len(val) > 80:
            val = val[:79] + "…"
        lines.append(f"{key}：{val}")
    return "\n".join(lines)


def _extractive_compact(
    overflow: list[dict[str, str]],
    prev: dict[str, str] | None = None,
) -> dict[str, str]:
    out = {key: ((prev or {}).get(key) or "无") for key in COMPACT_KEYS}
    blob = _turns_blob(overflow, 40)
    if not blob:
        return out
    if out["Progress"] == "无":
        out["Progress"] = blob
    else:
        merged = out["Progress"] + "；" + blob
        out["Progress"] = merged[-80:] if len(merged) > 80 else merged
    if out["Critical Context"] == "无":
        out["Critical Context"] = blob[:80]
    return out


def _llm_compact(
    llm: LLM,
    overflow: list[dict[str, str]],
    prev_text: str,
) -> dict[str, str] | None:
    if isinstance(llm, StubLLM) or not overflow:
        return None
    user = (
        "【旧摘要】\n" + ((prev_text or "").strip() or "无")
        + "\n【新溢出的对话】\n" + _turns_blob(overflow, 60)
    )
    try:
        data = llm.chat_json(
            [
                {"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            task="compact",
            thinking=False,
            temperature=0.2,
        )
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for key in COMPACT_KEYS:
        val = data.get(key)
        if val is None:
            val = data.get(key.replace(" & ", " and "))
        text = textutil.redact(str(val or "").strip()) or "无"
        out[key] = text[:80]
    return out


def compact_history(
    history: list[dict[str, str]] | None,
    *,
    llm: LLM | None = None,
    prev_summary: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """最近 6 轮原文；更早的压成 6.7 的固定小节，增量更新。"""
    items = list(history or [])
    prev = (prev_summary or "").strip()
    if len(items) <= KEEP_HISTORY:
        return prev, items
    older, recent = items[:-KEEP_HISTORY], items[-KEEP_HISTORY:]
    overflow = older[-2:] if prev else older
    sections = _llm_compact(llm, overflow, prev) if llm is not None else None
    if not sections:
        sections = _extractive_compact(overflow, _parse_summary(prev))
    return _render_summary(sections), recent


def _history_window(
    history: list[dict[str, str]] | None,
    *,
    llm: LLM | None = None,
    prev_summary: str = "",
) -> tuple[str, list[dict[str, str]]]:
    summary, recent = compact_history(
        history, llm=llm, prev_summary=prev_summary,
    )
    if not summary:
        return "", recent
    return "【这是之前的对话摘要】\n" + summary, recent


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": True,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _fn(
        "who_is",
        "查通讯录里某个人的身份、跟我的关系、我平时怎么称呼他。问某人是谁、跟某人什么关系时用。",
        {"name": {"type": "string", "description": "联系人名字或外号，例如 谢总、小胖"}},
        ["name"],
    ),
    _fn(
        "recent_with",
        "查我跟某个人最近几轮聊天。被问到最近跟他聊了什么、他最近说了什么时用。",
        {
            "name": {"type": "string", "description": "联系人名字或外号"},
            "limit": {"type": "integer", "description": "条数，1-12，不知道就填 8"},
        },
        ["name", "limit"],
    ),
    _fn(
        "search_chats",
        "在我的聊天记录里搜关键词，看我跟谁聊过、聊了什么。被问到某件事、某个话题时用。",
        {
            "keyword": {"type": "string", "description": "要搜的词，例如 打球、实习"},
            "contact": {"type": "string", "description": "可选，限定某个人；没有限定就填空字符串"},
            "limit": {"type": "integer", "description": "条数，1-12，不知道就填 8"},
        },
        ["keyword", "contact", "limit"],
    ),
    _fn(
        "my_facts",
        "查关于我本人的事实和偏好（学校、实习、作息、喜好）。被问到我喜欢什么、我的情况、平时怎样时必须用，不要只看人设卡那一行。",
        {"topic": {"type": "string", "description": "主题；没有具体主题就填空字符串"}},
        ["topic"],
    ),
]


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def run_tool(conn: sqlite3.Connection, call: ToolCall) -> dict[str, Any]:
    args = call.arguments or {}
    try:
        if call.name == "who_is":
            raw = str(args.get("name") or "")
            resolved = match_contact(conn, raw) or raw
            return kb.who_is(conn, resolved)
        if call.name == "recent_with":
            raw = str(args.get("name") or "")
            resolved = match_contact(conn, raw) or raw
            return kb.recent_with(conn, resolved, _as_int(args.get("limit"), 8, 1, 12))
        if call.name == "search_chats":
            raw = str(args.get("contact") or "")
            resolved = (match_contact(conn, raw) or raw) if raw else ""
            return kb.search_chats(
                conn, str(args.get("keyword") or ""),
                contact=resolved, limit=_as_int(args.get("limit"), 8, 1, 12),
            )
        if call.name == "my_facts":
            return kb.my_facts(conn, str(args.get("topic") or ""))
        return {"error": f"未知工具 {call.name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def tools_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("HA_AGENT_TOOLS", "1").strip().lower() not in {"0", "false", "no"}


def _thinking_on() -> bool:
    return os.environ.get("HA_AGENT_THINKING", "0").strip().lower() in {"1", "true", "yes"}


def _session_budget() -> float:
    try:
        return float(os.environ.get("HA_SESSION_BUDGET", "0.05"))
    except ValueError:
        return 0.05


def _is_fatal_llm(exc: BaseException) -> bool:
    text = str(exc)
    return any(tok in text for tok in (
        "HTTP 401", "HTTP 402", "HTTP 403",
        "Insufficient Balance", "invalid_api_key", "invalid api key",
    ))


def _tool_loop(
    llm: LLM,
    conn: sqlite3.Connection,
    messages: list[dict[str, Any]],
    *,
    max_hops: int,
) -> tuple[str, list[dict[str, Any]], Reply]:
    hops: int = 0
    trace: list[dict[str, Any]] = []
    started = time.time()
    last: Reply | None = None
    thinking = _thinking_on()
    while True:
        if time.time() - started > 60:
            if last and (last.text or "").strip():
                return last.text.strip(), trace, last
            raise LLMError("工具循环超时")
        last = llm.chat(
            messages, task="agent", temperature=0.8,
            tools=TOOLS, tool_choice="auto", thinking=thinking,
        )
        if not last.tool_calls or hops >= max_hops:
            text = (last.text or "").strip()
            if not text:
                raise LLMError("模型返回了空内容")
            return text, trace, last
        messages.append(last.as_message())
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        for call in last.tool_calls:
            key = (call.name, json.dumps(call.arguments or {}, ensure_ascii=False, sort_keys=True))
            if key not in cache:
                cache[key] = run_tool(conn, call)
                trace.append({"tool": call.name, "args": call.arguments})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(cache[key], ensure_ascii=False),
            })
        hops += 1
        if hops >= max_hops:
            messages.append({
                "role": "system",
                "content": "已经查够了，现在用你查到的东西回答，不要再调用工具。",
            })
            last = llm.chat(
                messages, task="agent", temperature=0.8,
                tools=TOOLS, tool_choice="none", thinking=thinking,
            )
            text = (last.text or "").strip()
            if not text:
                raise LLMError("模型返回了空内容")
            return text, trace, last


# 一条消息最多带几张图；历史里只保留最近一轮的图片，避免每轮重复上传烧 token
MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_HISTORY = 2


def image_blocks(images: list[str] | None) -> list[dict[str, Any]]:
    """把图片链接转成模型能吃的 content 块（OpenAI 兼容格式）。"""
    return [
        {"type": "image_url", "image_url": {"url": url}}
        for url in (images or [])
        if isinstance(url, str) and url.startswith("data:image/")
    ][:MAX_IMAGES_PER_MESSAGE]


def build_messages(
    conn: sqlite3.Connection,
    message: str,
    *,
    mode: str = MODE_PERSONA,
    partner: str = "",
    history: list[dict[str, str]] | None = None,
    images: list[str] | None = None,
    k_facts: int = 8,
    include_relations: bool = False,
    llm: LLM | None = None,
    prev_summary: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """组装 agent 的一轮对话。返回 (messages, 依据)。

    稳定段在前（人设、说话方式、人物索引），变化段在后。
    默认不把关系卡正文塞进提示词，留给 who_is，避免模型照抄。
    --no-tools 时 include_relations=True，走旧的备忘注入。
    """
    self_names = {"本人", "我", PERSONA_NAME, ASSISTANT_NAME, "林万达"}
    partner_raw = (partner or "").strip()
    is_self = partner_raw in self_names
    contact = "" if (not partner_raw or is_self) else partner_raw
    scene = kb.scene_for_contact(conn, contact) if contact else ""
    facts = kb.retrieve_facts(conn, message, contact=contact, k=k_facts)
    persona = kb.get_persona(conn)
    label, scene_label, tone = partner_label(conn, partner)
    if mode == MODE_ASSISTANT:
        label, scene_label, tone = "本人（林万达）", "助理", ""
    elif not contact and not is_self:
        label, scene_label, tone = "还没对上通讯录的人", "未识别", kb.tone_rule("不熟")

    parts: list[str] = []
    if persona:
        parts.append(kb.persona_block(persona))

    index = kb.person_index()
    if index:
        parts.append(index)

    if contact and mode != MODE_ASSISTANT:
        rel_block = kb.relation_prompt_block(contact, scene, brief=not include_relations)
        if rel_block:
            parts.append(rel_block)

    relation_facts = [f for f in facts if f.get("predicate") == "关系"]
    other_facts = [f for f in facts if f.get("predicate") != "关系"]
    if include_relations and relation_facts:
        parts.append(_relation_notes_block(relation_facts))
    if other_facts:
        block = "\n".join(f"- {f['predicate']}：{textutil.redact(f['object'])}" for f in other_facts)
        parts.append(f"【关于我的事实（别写错，也别编；细节不够就调 my_facts）】\n{block}")

    prefs = _relevant_preferences(conn, message)
    if prefs:
        parts.append("【跟这轮有关的我的偏好】\n"
                     + "\n".join(f"- {textutil.redact(p)}" for p in prefs))

    tier = ""
    if mode != MODE_ASSISTANT:
        if contact and scene:
            tier = kb.tier_for_scene(scene)
        elif contact:
            tier = kb.TIER_CLOSE if kb.relation_for(contact) else kb.TIER_PEER
        else:
            tier = stranger_tier(message)
        profile = kb.style_block(tier)
        if profile:
            parts.append(profile)

    if mode == MODE_ASSISTANT:
        parts.append(
            "【现在跟你说话的人】本人（林万达），你的用户。\n"
            "- 他不是你要模仿的对象：用什么语气跟他说话由你自己判断，"
            "简单的事就简短，复杂的事讲清楚，不必客套，也不必学他的口头禅。"
        )
    else:
        parts.append(
            f"【现在跟你说话的人】{label}"
            + (f"（场景：{scene_label}）" if scene_label else "")
            + f"\n- 在这个人面前我该有的语气：{tone}"
        )
        if not contact and not is_self:
            parts.append(
                "【这轮注意】通讯录没对上这个人。禁止问「你谁啊」，禁止说「不认识」"
                "「我不记得你」。按不熟的人直接回他这句话里问的事；"
                "对方报了名字就用这个称呼，不要质疑。"
            )
    said_text = textutil.redact(message).strip()
    attached = image_blocks(images)
    if attached:
        parts.append(
            f"【他发来 {len(attached)} 张图片】先看清图再回他；"
            "图里认不出的地方别硬猜，看不清就问他。"
            + (f"\n【他同时说】\n{said_text}" if said_text else "")
        )
    else:
        parts.append(f"【他说】\n{said_text}")
    said = textutil.normalize(message or "")
    if re.search(r"(是谁|什么关系|谁啊)", said):
        parts.append("【这轮注意】这是在问某个人是谁或什么关系：必须调用 who_is，不能只靠人物索引那一行。")
    elif re.search(r"(最近聊|最近说|聊了啥|聊了什么)", said):
        parts.append("【这轮注意】这是在问最近聊了什么：必须调用 recent_with。")
    elif re.search(r"(聊过|提起过)", said):
        parts.append("【这轮注意】这是在问某件事聊过没有：必须调用 search_chats。")
    elif re.search(r"(喜欢什么|平时喜欢|我的情况|平时怎样)", said):
        parts.append("【这轮注意】这是在问我的情况或偏好：必须调用 my_facts。")

    compacted, recent = compact_history(
        history, llm=llm, prev_summary=prev_summary,
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM}]
    if compacted:
        messages.append({
            "role": "system",
            "content": "【这是之前的对话摘要】\n" + compacted,
        })
    # 历史里只给最近一条用户消息恢复图片，更早的图不带（省 token，也避免重复上传）
    recent = list(recent)
    last_user = max((i for i, t in enumerate(recent) if t.get("role") != "agent"), default=-1)
    for i, turn in enumerate(recent):
        role = "assistant" if turn.get("role") == "agent" else "user"
        text = textutil.redact(str(turn.get("text", "")))
        history_images = image_blocks(turn.get("images")) if i == last_user else []
        history_images = history_images[:MAX_IMAGE_HISTORY]
        if history_images:
            content: Any = [{"type": "text", "text": text or "（图片）"}, *history_images]
        else:
            content = text
        messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": ([{"type": "text", "text": "\n\n".join(parts)}, *attached]
                    if attached else "\n\n".join(parts)),
    })

    evidence = {
        "persona": persona,
        "facts": other_facts,
        "tier": tier,
        "partner": label,
        "scene": scene_label,
        "tone": tone,
        "hit": ({"layer": "self", "label": ""} if mode == MODE_ASSISTANT
                else kb.hit_info(conn, contact, scene)),
        "tool_trace": [],
        "session_summary": compacted,
    }
    return messages, evidence


def reply(
    llm: LLM,
    conn: sqlite3.Connection,
    message: str,
    *,
    mode: str = MODE_PERSONA,
    partner: str = "",
    history: list[dict[str, str]] | None = None,
    images: list[str] | None = None,
    use_tools: bool | None = None,
    max_hops: int = 2,
    session_summary: str = "",
) -> dict[str, Any]:
    """回一句话。mode 决定是晚风（本人）还是早安（助理）。

    晚风认不出对方是谁时不再追问，直接按「不熟的人」那一档聊。
    默认开工具循环；HA_AGENT_TOOLS=0 或 use_tools=False 走旧路径。
    """
    resolved = ""
    if mode == MODE_PERSONA:
        resolved = infer_partner(conn, message) or (match_contact(conn, partner) or "")
        if resolved and kb.scene_for_contact(conn, resolved) == "忘记了":
            return {
                "agent": display_name(mode), "mode": mode, "need_identity": False,
                "partner": resolved, "text": FORGOT_REPLY, "model": "",
                "evidence": {
                    "partner": resolved, "scene": "忘记了",
                    "tone": kb.tone_rule("忘记了"),
                    "hit": kb.hit_info(conn, resolved, "忘记了"),
                    "facts": [], "style": [], "tool_trace": [],
                    "session_summary": session_summary or "",
                },
            }
    enabled = tools_enabled(use_tools)
    if enabled and llm.session_cost >= _session_budget():
        enabled = False
        budget_gate = True
    else:
        budget_gate = False

    messages, evidence = build_messages(
        conn, message, mode=mode, partner=resolved or partner,
        history=history, images=images, include_relations=not enabled,
        llm=llm, prev_summary=session_summary,
    )
    if mode == MODE_ASSISTANT:
        messages[0] = {"role": "system", "content": ASSISTANT_SYSTEM + NO_MARKDOWN}
    else:
        messages[0] = {"role": "system", "content": AGENT_SYSTEM + NO_MARKDOWN}
        if not resolved:
            evidence["scene"] = "未识别"
    if resolved:
        evidence["resolved_from_message"] = True
    if budget_gate:
        evidence["budget_gate"] = True

    out: Reply | None = None
    trace: list[dict[str, Any]] = []
    text = ""

    def _fail(msg: str, err: str) -> dict[str, Any]:
        evidence["tools_error"] = err[:200]
        evidence["tool_trace"] = trace
        return {
            "agent": display_name(mode), "mode": mode, "need_identity": False,
            "partner": resolved, "text": msg, "model": "", "evidence": evidence,
        }

    if enabled:
        try:
            text, trace, out = _tool_loop(llm, conn, messages, max_hops=max_hops)
        except Exception as exc:
            if _is_fatal_llm(exc):
                return _fail("模型这边暂时没法回：接口没额度或没通过鉴权。", str(exc))
            evidence["tools_fallback"] = True
            has_tool_msgs = any(m.get("role") == "tool" for m in messages)
            try:
                out = llm.chat(
                    messages, task="agent", temperature=0.8, thinking=False,
                    tools=TOOLS if has_tool_msgs else None,
                    tool_choice="none" if has_tool_msgs else None,
                )
                text = (out.text or "").strip()
            except Exception as exc2:
                return _fail("模型这边暂时没法回。", str(exc2))
    else:
        try:
            out = llm.chat(messages, task="agent", temperature=0.8, thinking=False)
            text = (out.text or "").strip()
        except Exception as exc:
            return _fail("模型这边暂时没法回。", str(exc))

    evidence["tool_trace"] = trace
    return {
        "agent": display_name(mode),
        "mode": mode,
        "need_identity": False,
        "partner": resolved,
        "text": text,
        "model": out.model if out else "",
        "evidence": evidence,
    }
