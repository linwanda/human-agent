"""模型网关。

统一走 OpenAI 兼容的 /chat/completions 协议，因此：
    * 现在用 DeepSeek（base_url = https://api.deepseek.com/v1）
    * 以后换通义、智谱、本地 Ollama、vLLM 只改 .env，不动业务代码
    * 论文里这就是「多模型可插拔」的设计点，也是消融实验的入口

只用标准库 urllib 发请求，不强制安装任何三方包。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


class LLMError(RuntimeError):
    pass


# 日志是多线程写入的，必须加锁，否则并发时会出现两行内容交织的坏行
_LOG_LOCK = threading.Lock()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""

    def as_message(self) -> dict[str, Any]:
        """回填进下一轮请求的 assistant 消息。

        带 tools 时，如果开了思考模式，必须把 reasoning_content 原样带回，
        否则 DeepSeek 直接 400。关思考时这段是空的，不会带上。
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.text if self.text else None,
        }
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return msg


def _extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON。容忍 ```json 围栏与前后废话。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = t.find(opener), t.rfind(closer)
        if start != -1 and end > start:
            candidate = t[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise LLMError(f"模型没有返回可解析的 JSON：{text[:200]}")


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for i, item in enumerate(message.get("tool_calls") or []):
        fn = item.get("function") or {}
        out.append(ToolCall(
            id=str(item.get("id") or f"call_{i}"),
            name=str(fn.get("name") or ""),
            arguments=_parse_arguments(fn.get("arguments")),
        ))
    return [c for c in out if c.name]


def _chat_url(base_url: str, *, strict: bool) -> str:
    """strict 模式走 DeepSeek 的 /beta；本地服务没有 beta，保持原地址。"""
    base = (base_url or "").rstrip("/")
    local = "127.0.0.1" in base or "localhost" in base
    if strict and not local:
        if base.endswith("/v1"):
            base = base[:-3] + "/beta"
        elif not base.endswith("/beta"):
            base = base + "/beta"
    return base + "/chat/completions"


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        settings.ensure_dirs()
        self.log_file: Path = settings.log_dir / "llm_calls.jsonl"
        self.session_cost: float = 0.0     # 本次进程已花费（美元），用于预算闸

    # ---------- 底层调用 ----------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        task: str = "generic",
        model: str | None = None,
        temperature: float = 0.8,
        json_mode: bool = False,
        thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Reply:
        base_url, api_key = self.s.endpoint_for(task)
        model = model or self.s.model_for(task)
        if not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost") and not api_key:
            raise LLMError(
                "缺少 API Key。把 DEEPSEEK_API_KEY 写进 .env（参考 .env.example），"
                "或加 --offline 用离线桩跑通流程。"
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # 思考模式不支持 temperature（传了不生效还可能干扰）。固化开思考，对话关思考。
        if thinking:
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["temperature"] = temperature
            if thinking is False:
                payload["thinking"] = {"type": "disabled"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        url = _chat_url(base_url, strict=bool(tools))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: Exception | None = None
        for attempt in range(max_retries):
            started = time.time()
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.s.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                latency = time.time() - started
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = (message.get("content") or "").strip()
                tool_calls = _parse_tool_calls(message)
                reasoning = (message.get("reasoning_content") or "").strip()
                if not text and not tool_calls:
                    # 偶尔会返回空内容（思考模式吃掉预算之类），当成一次失败重试
                    raise LLMError("模型返回了空内容")
                usage = data.get("usage") or {}
                reply = Reply(
                    text=text,
                    model=data.get("model", model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency=latency,
                    cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
                    cache_miss_tokens=usage.get("prompt_cache_miss_tokens", 0),
                    tool_calls=tool_calls,
                    reasoning_content=reasoning,
                )
                self._log(task, reply)
                return reply
            except urllib.error.HTTPError as exc:  # 429 / 5xx 值得重试
                detail = exc.read().decode("utf-8", "ignore")[:300]
                last_error = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code not in (429, 500, 502, 503, 504):
                    raise last_error
            except Exception as exc:  # 网络抖动
                last_error = exc
            time.sleep(1.5 * (2 ** attempt))
        raise LLMError(f"调用失败（重试 {max_retries} 次）：{last_error}")

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        reply = self.chat(messages, json_mode=True, **kwargs)
        return _extract_json(reply.text)

    # ---------- 记账 ----------
    def _log(self, task: str, reply: Reply) -> None:
        cache_hit = reply.cache_hit_tokens
        cache_miss = reply.cache_miss_tokens or max(0, reply.prompt_tokens - cache_hit)
        cost = self.s.estimate_cost(reply.model, cache_hit=cache_hit, cache_miss=cache_miss,
                                    output=reply.completion_tokens)
        self.session_cost += cost or 0.0
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "model": reply.model,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss,
            "latency": round(reply.latency, 3),
            "cost": round(cost, 6) if cost is not None else None,
            "tool_calls": len(reply.tool_calls),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _LOG_LOCK:
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(line)


_SAID = re.compile(r"【他说】\s*(.*)$", re.S)
_SMALL_TALK = re.compile(
    r"^(在吗|在不在|嗯+|哦+|哈+|好+|ok+|okay|你好|嗨|嘿|早|晚上好)[\s!！。.~～]*$",
    re.I,
)


def _last_said(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "")
        match = _SAID.search(content)
        return (match.group(1) if match else content).strip()
    return ""


class StubLLM(LLM):
    """离线桩：不需要 Key 就能把整条流水线跑通，也方便写单元测试。

    它不做任何真实的语言理解，只按任务类型返回结构化占位结果。
    带 tools 时按关键词决定调不调，用来验收循环，不模拟「像人说话」。
    """

    _seq = 0

    def chat(self, messages: list[dict[str, Any]], *, task: str = "generic", **kwargs: Any) -> Reply:
        tools = kwargs.get("tools")
        user = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        if task == "extract":
            facts = [{"subject": "我", "predicate": "提及内容", "object": user[-30:].strip(), "confidence": 0.3}]
            text = json.dumps({"facts": facts, "preferences": []}, ensure_ascii=False)
            return Reply(text=text, model="stub", latency=0.0)
        if task == "agent" and tools:
            if any(m.get("role") == "tool" for m in messages):
                return Reply(text="离线桩：我在，你说。", model="stub", latency=0.0)
            if kwargs.get("tool_choice") == "none":
                return Reply(text="离线桩：我在，你说。", model="stub", latency=0.0)
            said = _last_said(messages)
            call = self._stub_tool(said)
            if call:
                return Reply(text="", model="stub", latency=0.0, tool_calls=[call])
            return Reply(text="离线桩：我在，你说。", model="stub", latency=0.0)
        if task == "agent":
            return Reply(text="离线桩：我在，你说。", model="stub", latency=0.0)
        return Reply(text="离线桩输出", model="stub", latency=0.0)

    def _stub_tool(self, said: str) -> ToolCall | None:
        text = (said or "").strip()
        if not text or _SMALL_TALK.match(text):
            return None
        StubLLM._seq += 1
        call_id = f"call_stub_{StubLLM._seq}"
        if re.search(r"(最近聊了|聊了什么|聊了啥|他说了什么|最近说了)", text):
            name = "谢总"
            hit = re.search(r"(?:跟|和)([^，。？?！!\s]{1,12})最近", text)
            if hit:
                name = hit.group(1)
            return ToolCall(call_id, "recent_with", {"name": name, "limit": 8})
        if re.search(r"(聊过|搜过|提起过|有没有聊)", text):
            keyword = "打球"
            hit = re.search(r"聊过\s*([^，。？?！!\s]{1,12})", text)
            if hit:
                keyword = hit.group(1).rstrip("吗么")
            return ToolCall(call_id, "search_chats",
                            {"keyword": keyword, "contact": "", "limit": 8})
        if re.search(r"(喜欢什么|有啥偏好|我的情况|平时喜欢)", text):
            return ToolCall(call_id, "my_facts", {"topic": text[:12]})
        if re.search(r"(是谁|什么关系|谁啊)", text):
            name = "谢总"
            hit = re.search(r"([^，。？?！!\s]{1,12})是谁", text)
            if hit:
                name = hit.group(1).lstrip("我跟和")
            return ToolCall(call_id, "who_is", {"name": name})
        return None

    def _log(self, task: str, reply: Reply) -> None:
        return


def get_llm(settings: Settings, offline: bool | None = None) -> LLM:
    if offline is None:
        offline = os.environ.get("HA_OFFLINE", "0").strip() in {"1", "true", "yes"}
    return StubLLM(settings) if offline else LLM(settings)


def summarize_calls(settings: Settings) -> dict:
    """汇总调用日志，并按当前平均用量给出后续任务的费用预估（美元）。"""
    path = settings.log_dir / "llm_calls.jsonl"
    if not path.exists():
        return {"calls": 0, "note": "还没有调用记录"}

    by_task: dict[str, dict[str, float]] = {}
    by_model: dict[str, dict[str, float]] = {}
    total_cost = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue          # 并发写入偶发的坏行直接跳过，不影响汇总
        cost = row.get("cost") or 0.0
        total_cost += cost
        for table, key in ((by_task, row["task"]), (by_model, row["model"])):
            item = table.setdefault(key, {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0})
            item["calls"] += 1
            item["prompt"] += row["prompt_tokens"]
            item["completion"] += row["completion_tokens"]
            item["cost"] += cost

    def avg(table: dict, task: str) -> tuple[int, int, float]:
        item = table.get(task)
        if not item or not item["calls"]:
            return 0, 0, 0.0
        return (int(item["prompt"] / item["calls"]), int(item["completion"] / item["calls"]),
                item["cost"] / item["calls"])

    extract_prompt, extract_out, extract_unit = avg(by_task, "extract")
    chat_prompt, chat_out, chat_unit = avg(by_task, "agent")

    # 用当前平均用量外推：假设全部按平价时段计费
    def project(count: int, prompt: int, out: int, model: str) -> float | None:
        if not prompt:
            return None
        return settings.estimate_cost(model, cache_miss=prompt * count, output=out * count,
                                      when=datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc))

    projections = {
        "抽取 1 万条": project(10_000, extract_prompt, extract_out, settings.model_for("extract")),
        "抽取 10 万条": project(100_000, extract_prompt, extract_out, settings.model_for("extract")),
        "每天聊 50 句": project(50, chat_prompt, chat_out, settings.model_for("agent")),
        "每天聊 200 句": project(200, chat_prompt, chat_out, settings.model_for("agent")),
    }
    return {
        "calls": sum(int(v["calls"]) for v in by_task.values()),
        "total_cost_usd": round(total_cost, 6),
        "by_task": {k: {"calls": int(v["calls"]), "avg_prompt": int(v["prompt"] / v["calls"]),
                        "avg_completion": int(v["completion"] / v["calls"]),
                        "cost_usd": round(v["cost"], 6)} for k, v in by_task.items()},
        "by_model": {k: {"calls": int(v["calls"]), "cost_usd": round(v["cost"], 6)}
                     for k, v in by_model.items()},
        "projection_usd": {k: (round(v, 4) if v is not None else None) for k, v in projections.items()},
    }
