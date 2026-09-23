"""配置与密钥管理。

原则：任何密钥都不写进代码，只从环境变量或根目录的 .env 读取。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = ROOT / ".env"


def load_env(path: Path | None = None) -> None:
    """极简 .env 解析：KEY=VALUE，忽略 # 注释，不覆盖已存在的环境变量。"""
    path = path or DEFAULT_ENV_FILE
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 任务 -> 模型档位。改这里就能换模型分工，不用动业务代码。
TASK_ROUTES: dict[str, str] = {
    "agent": "main",        # 晚风 / 早安连续对话（在线，高频）
    "rewrite": "main",      # 润色 / 改写（在线，中频）
    "compact": "main",      # 长聊摘要（在线，仅历史溢出时）
    "extract": "main",      # 事实与偏好抽取（离线批处理）
    "interview": "main",    # 访谈提问生成（离线，低频）
    "judge": "reasoner",    # 评测裁判（离线，低频，需更强推理）
}


@dataclass
class Settings:
    root: Path = ROOT
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "kb.sqlite3")
    log_dir: Path = field(default_factory=lambda: ROOT / "data" / "logs")
    export_dir: Path = field(default_factory=lambda: ROOT / "data" / "train")
    eval_dir: Path = field(default_factory=lambda: ROOT / "data" / "eval")
    people_dir: Path = field(default_factory=lambda: ROOT / "data" / "people")

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model_main: str = "deepseek-flash"
    model_reasoner: str = "deepseek-v4-pro"
    timeout: int = 120

    # 隐私开关：为真时只允许调用本地模型（本地服务同样是 OpenAI 兼容接口）
    privacy_mode: bool = False
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "qwen2.5:7b-instruct"

    # 离线准备参数
    context_turns: int = 6          # 每条本人发言向前保留的上下文条数
    min_self_chars: int = 2         # 去掉标点表情后短于该长度的发言视为低信息量
    self_name: str = ""             # 聊天记录里代表「你」的昵称，如 晚风

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.log_dir, self.export_dir, self.eval_dir, self.people_dir):
            d.mkdir(parents=True, exist_ok=True)

    def model_for(self, task: str) -> str:
        level = TASK_ROUTES.get(task, "main")
        if self.privacy_mode:
            return self.local_model
        return self.model_reasoner if level == "reasoner" else self.model_main

    def endpoint_for(self, task: str) -> tuple[str, str]:
        """返回 (base_url, api_key)。隐私模式下切到本地服务。"""
        if self.privacy_mode:
            return self.local_base_url, os.environ.get("HA_LOCAL_API_KEY", "local")
        return self.base_url, self.api_key

    def pricing(self) -> dict:
        """价目表（美元/百万 token），来源与更新时间见 pricing.json 里的说明。"""
        path = self.root / "pricing.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def estimate_cost(
        self,
        model: str,
        *,
        cache_hit: int = 0,
        cache_miss: int = 0,
        output: int = 0,
        when: datetime | None = None,
    ) -> float | None:
        """按官网价目表算一次调用的花费（美元）。

        高峰价是平价的两倍。高峰为 UTC 周一至周五 01:00-04:00 与 06:00-10:00，
        折成北京时间是工作日的 09:00-12:00 和 14:00-18:00。
        也就是说批量任务放到晚上跑，价格直接减半。
        """
        table = self.pricing().get("models", {}).get(model)
        if not table:
            return None
        when = when or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        utc = when.astimezone(timezone.utc)
        peak = utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)
        key = "peak" if peak else "off"
        return (
            cache_hit / 1_000_000 * table.get("cache_hit", {}).get(key, 0.0)
            + cache_miss / 1_000_000 * table.get("cache_miss", {}).get(key, 0.0)
            + output / 1_000_000 * table.get("output", {}).get(key, 0.0)
        )


def get_settings(env_file: Path | None = None) -> Settings:
    load_env(env_file)
    s = Settings()
    s.api_key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("HA_API_KEY", "")
    s.base_url = os.environ.get("HA_BASE_URL", s.base_url)
    s.model_main = os.environ.get("HA_MODEL_MAIN", s.model_main)
    s.model_reasoner = os.environ.get("HA_MODEL_REASONER", s.model_reasoner)
    s.privacy_mode = os.environ.get("HA_PRIVACY_MODE", "0").strip() in {"1", "true", "True", "yes"}
    s.local_base_url = os.environ.get("HA_LOCAL_BASE_URL", s.local_base_url)
    s.local_model = os.environ.get("HA_LOCAL_MODEL", s.local_model)
    if os.environ.get("HA_DB_PATH"):
        s.db_path = Path(os.environ["HA_DB_PATH"])
    if os.environ.get("HA_CONTEXT_TURNS"):
        s.context_turns = int(os.environ["HA_CONTEXT_TURNS"])
    if os.environ.get("HA_MIN_SELF_CHARS"):
        s.min_self_chars = int(os.environ["HA_MIN_SELF_CHARS"])
    s.self_name = os.environ.get("HA_SELF_NAME", s.self_name)
    return s
