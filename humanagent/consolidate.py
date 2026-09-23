"""把近圈聊天记录固化成「我眼里的他」。

离线批处理，不依赖对话工具。素材由 kb.sample_person 取，
模型只写正文和依据，front-matter 由代码写。
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import db, kb, textutil
from .config import ROOT, Settings
from .llm import LLM, LLMError, StubLLM

PEOPLE_DIR = ROOT / "data" / "people"
FAILURES_PATH = PEOPLE_DIR / "_failures.json"
_FAIL_LOCK = threading.Lock()

CONSOLIDATE_SYSTEM = """你在帮一个人整理「我眼里的某个人」：把原始聊天记录消化成他本人视角的一段理解，供他以后的 AI 分身引用。

硬要求：
1. 第一人称是资料主人本人，不是助手。
2. 只依据材料，不许编；材料看不出的不要写，材料不足就明说。
3. 给判断，不要罗列；数字只有在它本身说明问题时才提。
4. 不许出现字段名（关系/话题/状态），不许照抄原话，不许列群名清单。
5. 不许写"关系很好"这种空话——说清楚好在哪、在什么场景下会找他/她。
6. 一段话，不要小标题。要写出这个人在我生活里的位置：什么事会找对方、对方怎么对我、不同时期有没有变化。覆盖至少两个场景（比如办事/情绪/玩/日常），不要被某一类闲聊带成单面标签。
7. 聊天记录里的错别字、省略、口误按原意理解，不要当成性格，也不要写进依据里纠错。
8. 对方性别从聊天、称呼和备忘判断；备忘与聊天冲突时以聊天为准，但不要无根据地一律写成「他」。
9. 220 字以内。
"""

_FIELD_LINE = re.compile(r"^-?\s*(关系|话题|状态|互动统计|常聊|置信度|更新时间)\s*[：:]")


def _oneliner_from(body: str) -> str:
    text = (body or "").replace("\n", " ").strip()
    if not text:
        return ""
    for sep in ("。", "！", "？", ".", "!", "?"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    text = textutil.redact(text)
    if len(text) <= 40:
        return text
    chunk = text[:40]
    for sep in ("，", "；", ",", "、"):
        cut = chunk.rfind(sep)
        if cut >= 12:
            return chunk[:cut]
    return chunk


def _split_model_output(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("markdown"):
            text = text[8:].strip()
    if text.startswith("---"):
        rest = text[3:]
        end = rest.find("\n---")
        if end != -1:
            text = rest[end + 4:].strip()
    evidence = ""
    if "## 依据" in text:
        body, rest = text.split("## 依据", 1)
        if "## 历史" in rest:
            rest = rest.split("## 历史", 1)[0]
        evidence = "## 依据\n" + rest.strip()
        text = body
    cleaned: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# ") or _FIELD_LINE.match(s):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip(), evidence.strip()


def _render(meta: dict[str, str], body: str, evidence: str, history: str = "") -> str:
    order = ("name", "scene", "oneliner", "stats", "model", "updated",
             "confidence", "source_hash", "stale")
    lines = ["---"]
    for key in order:
        lines.append(f"{key}: {meta.get(key, '')}")
    lines += ["---", "", body.strip(), ""]
    if evidence:
        if not evidence.startswith("## "):
            evidence = "## 依据\n" + evidence
        lines.append(evidence.rstrip())
        lines.append("")
    if history.strip():
        lines.append("## 历史")
        lines.append(history.strip())
        lines.append("")
    return "\n".join(lines)


def _record_failure(name: str, error: str) -> None:
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    with _FAIL_LOCK:
        rows: list[dict[str, str]] = []
        if FAILURES_PATH.exists():
            try:
                rows = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                rows = []
        rows.append({
            "name": name,
            "error": error[:300],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        FAILURES_PATH.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_stub(sample: dict[str, Any], card: dict[str, str]) -> Path:
    name = sample["name"]
    who = (card.get("who") or "").split("。")[0].split("；")[0].strip()
    meta = {
        "name": name,
        "scene": sample.get("scene") or "",
        "oneliner": textutil.redact(who)[:40] or "记录太少，只进索引",
        "stats": sample.get("stats") or "记录很少",
        "model": "none",
        "updated": time.strftime("%Y-%m-%d"),
        "confidence": "low",
        "source_hash": sample["source_hash"],
        "stale": "false",
    }
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = kb._people_path(name)
    path.write_text(_render(meta, "记录太少，暂不做判断。", ""), encoding="utf-8")
    return path


def _old_history(existing: dict[str, str]) -> str:
    old = (existing.get("understanding") or "").strip()
    if not old or old == "记录太少，暂不做判断。":
        extra = ""
        body = existing.get("body") or ""
        if "## 历史" in body:
            extra = body.split("## 历史", 1)[1].strip()
        return extra
    stamp = existing.get("updated") or ""
    block = f"{stamp} {old}".strip() if stamp else old
    extra = ""
    body = existing.get("body") or ""
    if "## 历史" in body:
        extra = body.split("## 历史", 1)[1].strip()
    return (block + ("\n\n" + extra if extra else "")).strip()


def consolidate_one(
    conn,
    llm: LLM,
    settings: Settings,
    name: str,
    model: str | None = None,
) -> dict[str, Any]:
    sample = kb.sample_person(conn, name)
    resolved = sample["name"]
    card = kb.relation_for(resolved) or {}
    path = kb._people_path(resolved)
    existing = kb.parse_people_md(path.read_text(encoding="utf-8")) if path.exists() else {}

    if (existing.get("source_hash") == sample["source_hash"]
            and existing.get("stale") != "true"
            and "## 依据" in (existing.get("body") or "")):
        return {"name": resolved, "skipped": "source_hash 未变", "path": str(path)}

    if sample.get("too_few"):
        written = _write_stub(sample, card)
        return {
            "name": resolved, "path": str(written), "skipped": "记录太少，只进索引",
            "confidence": "low", "usable": sample.get("usable", 0),
        }

    if isinstance(llm, StubLLM):
        return {"name": resolved, "skipped": "离线桩不固化"}

    chosen = model or settings.model_main
    think = "v4-pro" in chosen or chosen == settings.model_reasoner
    user = (
        f"【这个人】{resolved}（括号里是微信备注的日期，不是生日）\n"
        f"【大类】{sample.get('scene') or '未分类'}\n"
        + (f"【长期备忘（只校正是谁、性别、长期位置，禁止照抄；细节必须来自下面的聊天）】{card['who']}\n"
           if card.get("who") else "")
        + f"【素材统计】{sample['stats']}\n"
        f"【我和对方的聊天片段（按时间，我发的标「我」，对方发的标「对方」；错别字按意思读）】\n"
        + "\n".join(sample["lines"])
        + f"\n\n按上面的要求，写出我眼里的{resolved}。"
        "正文写完后必须另起一节 ## 依据，列 3 到 5 条，每条一句，带来自不同时期或不同场景的时间和原话，方便核对你没有编。"
    )
    messages = [
        {"role": "system", "content": CONSOLIDATE_SYSTEM},
        {"role": "user", "content": user},
    ]
    last_error = ""
    out_text = ""
    used_model = chosen
    for attempt in range(3):
        try:
            reply = llm.chat(
                messages, task="judge", model=chosen,
                thinking=think,
            )
            out_text = (reply.text or "").strip()
            used_model = reply.model or chosen
            if out_text and "## 依据" in out_text:
                break
            last_error = "缺依据" if out_text else "模型返回了空内容"
            out_text = ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:240]
            if isinstance(exc, LLMError) and any(
                tok in str(exc) for tok in ("HTTP 401", "HTTP 402", "Insufficient Balance")
            ):
                break
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    if not out_text:
        _record_failure(resolved, last_error or "空内容")
        return {"name": resolved, "failed": last_error or "空内容"}

    body, evidence = _split_model_output(out_text)
    if not body:
        _record_failure(resolved, "解析后正文为空")
        return {"name": resolved, "failed": "解析后正文为空"}
    if "## 依据" not in evidence:
        _record_failure(resolved, "缺依据")
        return {"name": resolved, "failed": "缺依据"}
    oneliner = _oneliner_from(body)
    if not oneliner:
        oneliner = ((card.get("who") or "").split("。")[0])[:40]
    meta = {
        "name": resolved,
        "scene": sample.get("scene") or "",
        "oneliner": oneliner,
        "stats": sample["stats"],
        "model": used_model,
        "updated": time.strftime("%Y-%m-%d"),
        "confidence": "high",
        "source_hash": sample["source_hash"],
        "stale": "false",
    }
    history = _old_history(existing) if existing.get("understanding") else ""
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(meta, body, evidence, history), encoding="utf-8")
    return {
        "name": resolved, "path": str(path), "chars": len(body),
        "model": used_model, "usable": sample["usable"],
        "lines": len(sample["lines"]), "source_hash": sample["source_hash"],
    }


def _targets(top: int, names: list[str] | None) -> list[str]:
    if names:
        return names
    cards = list(kb._relations().keys())
    return cards[: max(1, top)]


def consolidate_top(
    llm: LLM,
    conn,
    settings: Settings | None = None,
    *,
    top: int = 89,
    names: list[str] | None = None,
    model: str | None = None,
    workers: int = 4,
    stale_only: bool = False,
    sample_run: bool = False,
    reasoner: bool = False,
) -> dict[str, Any]:
    settings = settings or Settings()
    if reasoner and not model:
        model = settings.model_reasoner
    if sample_run and not model:
        model = settings.model_reasoner
    old_timeout = settings.timeout
    settings.timeout = max(old_timeout, 180)
    try:
        if stale_only:
            report = check_people(conn)
            names = report.get("stale") or []
            if not names:
                return {"done": 0, "skipped": 0, "failed": 0, "note": "没有 stale"}
        targets = _targets(top, names)
        done, skipped, failed = [], [], []

        def model_for(index: int) -> str | None:
            if model:
                return model
            if index < 20:
                return settings.model_reasoner
            return settings.model_main

        db_path = settings.db_path

        def job(item: tuple[int, str]) -> dict[str, Any]:
            index, person = item
            local = db.connect(db_path)
            try:
                return consolidate_one(
                    local, llm, settings, person, model=model_for(index),
                )
            finally:
                local.close()

        workers = max(1, min(int(workers or 1), len(targets)))
        if workers == 1 or len(targets) == 1:
            results = [job((i, n)) for i, n in enumerate(targets)]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(job, (i, n)): n for i, n in enumerate(targets)}
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        person = futs[fut]
                        err = f"{type(exc).__name__}: {exc}"[:200]
                        _record_failure(person, err)
                        results.append({"name": person, "failed": err})

        for result in results:
            if result.get("failed"):
                failed.append(result)
            elif result.get("skipped"):
                skipped.append(result)
            else:
                done.append(result)
        return {
            "done": len(done), "skipped": len(skipped), "failed": len(failed),
            "cost_usd": round(llm.session_cost, 6),
            "files": done, "skipped_detail": skipped, "failed_detail": failed,
        }
    finally:
        settings.timeout = old_timeout


def list_people() -> list[dict[str, str]]:
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(PEOPLE_DIR.glob("*.md")):
        out.append({
            "name": path.stem,
            "path": str(path),
            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
        })
    return out


def read_people(name: str) -> str:
    doc = kb.read_people_doc(name)
    if doc.get("body") or doc.get("understanding"):
        path = kb._people_path(doc.get("name") or name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        for item in PEOPLE_DIR.glob("*.md"):
            if name in item.stem:
                return item.read_text(encoding="utf-8")
    path = kb._people_path(name)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def check_people(conn=None) -> dict[str, Any]:
    """素材变了标 stale；缺字段列出来。"""
    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in PEOPLE_DIR.glob("*.md") if not p.name.startswith("_")]
    issues: list[str] = []
    stale: list[str] = []
    close_conn = False
    if conn is None:
        from .config import get_settings
        conn = db.connect(get_settings().db_path)
        close_conn = True
    try:
        for path in files:
            doc = kb.parse_people_md(path.read_text(encoding="utf-8"))
            name = doc.get("name") or path.stem
            for key in ("name", "scene", "oneliner", "stats", "model",
                        "updated", "confidence", "source_hash", "stale"):
                if not doc.get(key):
                    issues.append(f"{name}：缺 {key}")
            sample = kb.sample_person(conn, name)
            hash_changed = doc.get("source_hash") != sample["source_hash"]
            if hash_changed or doc.get("stale") == "true":
                stale.append(name)
                if hash_changed and doc.get("stale") != "true":
                    raw = path.read_text(encoding="utf-8")
                    path.write_text(
                        re.sub(r"^stale:\s*\S+", "stale: true", raw, count=1, flags=re.M),
                        encoding="utf-8",
                    )
            body = doc.get("understanding") or ""
            if "生日" in body and re.search(r"\d+\.\d+", name):
                issues.append(f"{name}：正文写了生日，括号里的数字是备注")
        return {"files": len(files), "stale": stale, "issues": issues}
    finally:
        if close_conn:
            conn.close()
