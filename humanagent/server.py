"""本地网页界面 + JSON 接口。

晚风 Agent 的主入口：浏览器里连续对话（晚风 / 早安）。
同一套内核也可被外部工作流通过 HTTP 调用。

启动：python cli.py web --port 8765
"""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import agent, db, kb
from .config import Settings
from .llm import get_llm, summarize_calls

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 图片只收浏览器给的 data URL；单张体积（字符数）和条数都设上限，防止一次塞爆
IMAGE_PREFIX = "data:image/"
MAX_IMAGES = 4
MAX_IMAGE_CHARS = 8_000_000


def _clean_images(raw: object) -> list[str]:
    """把前端传来的图片挑干净：只留 data:image/... 开头的字符串。"""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("url") or item.get("data") or ""
        if not isinstance(item, str):
            continue
        if not item.startswith(IMAGE_PREFIX) or len(item) > MAX_IMAGE_CHARS:
            continue
        out.append(item)
        if len(out) >= MAX_IMAGES:
            break
    return out


def _contacts(conn: sqlite3.Connection, limit: int = 60) -> list[dict]:
    rows = conn.execute(
        """SELECT c.name AS name, c.is_group AS is_group, MAX(e.scene) AS scene,
                  SUM(CASE WHEN e.role='self' THEN 1 ELSE 0 END) AS self_n,
                  COUNT(*) AS total
           FROM episodes e JOIN conversations c ON c.id = e.conv_id
           GROUP BY c.id ORDER BY self_n DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [{"name": r["name"], "group": bool(r["is_group"]), "scene": r["scene"] or "",
             "self_n": r["self_n"], "total": r["total"]} for r in rows]


def _facts(conn: sqlite3.Connection, limit: int = 400) -> list[dict]:
    rows = conn.execute(
        """SELECT predicate, object, confidence FROM facts
           WHERE status='active' ORDER BY predicate, confidence DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [{"predicate": r["predicate"], "object": r["object"],
             "confidence": r["confidence"]} for r in rows]


def _overview(conn: sqlite3.Connection, settings: Settings) -> dict:
    counts = db.table_counts(conn)
    return {
        "counts": counts,
        "scenes": dict(kb.scenes_of(conn)),
        "persona": kb.get_persona(conn),
        "preferences": [{"domain": r["domain"], "statement": r["statement"],
                         "polarity": r["polarity"]}
                        for r in conn.execute(
                            "SELECT domain, statement, polarity FROM preferences "
                            "WHERE status='active' ORDER BY domain LIMIT 300")],
        "cost": summarize_calls(settings),
    }


def build_handler(settings: Settings, offline: bool | None):
    llm = get_llm(settings, offline=offline)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):        # 静音，保持终端干净
            return

        # ---------- 工具 ----------
        def _send(self, code: int, body: str | bytes, ctype: str) -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj: dict, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def _conn(self) -> sqlite3.Connection:
            return db.connect(settings.db_path)

        # ---------- 路由 ----------
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                page = WEB_DIR / "index.html"
                if not page.exists():
                    self._send(500, "缺少 web/index.html", "text/plain; charset=utf-8")
                    return
                self._send(200, page.read_text(encoding="utf-8"), "text/html; charset=utf-8")
                return
            if not settings.db_path.exists():
                self._json({"error": "数据库不存在，先运行 init"}, 400)
                return
            conn = self._conn()
            try:
                if path == "/api/overview":
                    self._json(_overview(conn, settings))
                elif path == "/api/contacts":
                    self._json({"contacts": _contacts(conn)})
                elif path == "/api/facts":
                    self._json({"facts": _facts(conn)})
                else:
                    self._json({"error": "not found"}, 404)
            finally:
                conn.close()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            payload = self._body()
            if not settings.db_path.exists():
                self._json({"error": "数据库不存在，先运行 init"}, 400)
                return
            conn = self._conn()
            try:
                if path == "/api/chat":
                    message = (payload.get("message") or "").strip()
                    images = _clean_images(payload.get("images"))
                    if not message and not images:
                        self._json({"error": "消息不能为空"}, 400)
                        return
                    self._json(agent.reply(
                        llm, conn, message,
                        mode=(payload.get("mode") or agent.MODE_PERSONA),
                        partner=(payload.get("partner") or ""),
                        history=payload.get("history") or [],
                        images=images,
                        session_summary=str(payload.get("summary") or ""),
                        use_tools=None if payload.get("tools") is None else bool(payload.get("tools")),
                    ))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:                       # 任何异常都回给前端，别静默失败
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            finally:
                conn.close()

    return Handler


def serve(settings: Settings, port: int = 8765, host: str = "127.0.0.1",
          offline: bool | None = None) -> None:
    handler = build_handler(settings, offline)
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        err = getattr(exc, "winerror", None) or getattr(exc, "errno", None)
        if err in {98, 10048}:
            print(f"端口 {port} 已被占用，没法再开一次。")
            print(f"若晚风已经在跑，浏览器打开 http://{host}:{port} 即可。")
            print("若要重启：关掉旧的黑色窗口，或再双击一次「启动晚风.bat」（会先关掉旧进程）。")
            raise SystemExit(2) from exc
        raise
    print(f"网页界面已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
