"""命令行入口。

常用流程：
    python cli.py init
    python cli.py ingest examples/sample_chat.csv --self-name 我
    python cli.py chat "在吗，晚上一起吃饭吗" --contact 谢总
    python cli.py web
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from humanagent import agent, db, healthcheck, ingest, kb
from humanagent.config import get_settings
from humanagent.llm import get_llm, summarize_calls


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="晚风 Agent：连续对话的本人原型 agent")
    parser.add_argument("--db", help="覆盖数据库路径")
    parser.add_argument("--offline", action="store_true", help="用离线桩，不调真实模型")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="建库")
    sub.add_parser("doctor", help="环境自检")
    sub.add_parser("stats", help="知识库统计")

    p = sub.add_parser("health", help="数据体检：只统计不写库")
    p.add_argument("--dir", required=True, help="导出目录，会递归查找 _messages.csv")
    p.add_argument("--out", default="", help="报告输出目录，默认 data/report")

    p = sub.add_parser("rescan-scenes", help="按最新场景映射重刷已有记录的场景标签")
    p.add_argument("--scene", default="日常", help="没匹配上时用的默认场景")

    p = sub.add_parser("web", help="启动晚风 Agent 网页（晚风 / 早安）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")

    p = sub.add_parser("ingest", help="导入聊天记录")
    p.add_argument("path")
    p.add_argument("--self-name", help="你的昵称（导出文件里说话人字段的取值），默认读 HA_SELF_NAME")
    p.add_argument("--include-groups", action="store_true")
    p.add_argument("--mask", action="store_true", help="对长数字串脱敏")
    p.add_argument("--scene", default="", help="默认场景标签，如 日常")

    p = sub.add_parser("persona", help="人设卡：直接进入每次生成的提示词")
    p.add_argument("--set", action="append", default=[], metavar="键=值", help="可重复，如 --set 昵称=晚风")
    p.add_argument("--show", action="store_true")

    p = sub.add_parser("fact", help="手工录入一条事实")
    p.add_argument("--predicate", required=True)
    p.add_argument("--object", required=True, dest="object_value")
    p.add_argument("--confidence", type=float, default=0.95)

    p = sub.add_parser("pref", help="手工录入一条偏好")
    p.add_argument("--domain", default="其他")
    p.add_argument("--statement", required=True)
    p.add_argument("--polarity", default="like", choices=["like", "dislike", "neutral"])

    p = sub.add_parser("extract", help="抽取事实与偏好")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--workers", type=int, default=4, help="并发请求数，1 表示顺序执行")
    p.add_argument("--max-cost", type=float, default=1.0, help="本次花费上限（美元），默认 1 美元")

    p = sub.add_parser("chat", help="晚风 / 早安连续对话（主入口，直接回一句）")
    p.add_argument("message")
    p.add_argument("--mode", choices=["persona", "assistant"], default="persona",
                   help="persona=晚风（本人），assistant=早安（助理）")
    p.add_argument("--contact", default="", help="对方是谁（晚风模式会按关系切语气）")
    p.add_argument("--no-tools", action="store_true", help="关掉工具循环，走旧的单次调用")
    p.add_argument("--trace", action="store_true", help="把工具调用打到 stderr")

    p = sub.add_parser("consolidate", help="把近圈固化成 data/people/*.md")
    p.add_argument("--top", type=int, default=89, help="按薄卡顺序固化前 N 人")
    p.add_argument("--name", action="append", default=[], help="只固化这些人，可重复")
    p.add_argument("--sample", nargs="+", default=None, metavar="NAME",
                   help="只跑这些人。先审 3 个：谢总（12.22） 空想号MEK(小胖) 宝宝（9.8）")
    p.add_argument("--model", default="", help="覆盖模型，如 deepseek-v4-pro")
    p.add_argument("--reasoner", action="store_true", help="等同 --model 强档")
    p.add_argument("--check", action="store_true", help="只检查已有文件，不调模型")
    p.add_argument("--stale", action="store_true", help="只重跑 source_hash 变了的")
    p.add_argument("--workers", type=int, default=4, help="并发数，默认 4")

    p = sub.add_parser("people", help="查看固化后的「我眼里的他」")
    p.add_argument("--name", default="", help="看某个人")
    p.add_argument("--list", action="store_true")

    sub.add_parser("cost", help="调用花费汇总与后续任务费用预估")

    args = parser.parse_args(argv)
    settings = get_settings()
    if args.db:
        settings.db_path = Path(args.db)
    llm = get_llm(settings, offline=args.offline)

    if args.cmd == "init":
        path = db.init_db(settings.db_path)
        _print({"db": str(path), "status": "created"})
        return 0

    if args.cmd == "doctor":
        info = {"python": sys.version.split()[0], "db": str(settings.db_path),
                "db_exists": settings.db_path.exists(),
                "has_api_key": bool(settings.api_key), "privacy_mode": settings.privacy_mode,
                "model_main": settings.model_main, "model_reasoner": settings.model_reasoner,
                "offline": args.offline}
        if settings.db_path.exists():
            import sqlite3
            conn = db.connect(settings.db_path)
            info["counts"] = db.table_counts(conn)
            info["fts5"] = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='episodes_fts'").fetchone())
            conn.close()
        _print(info)
        return 0

    if args.cmd == "health":
        root = Path(args.dir)
        files = sorted(root.rglob("_messages.csv"))
        if not files:
            print(f"在 {root} 下没有找到 _messages.csv", file=sys.stderr)
            return 2
        result = healthcheck.scan(files, settings.min_self_chars)
        out_dir = Path(args.out) if args.out else settings.root / "data" / "report"
        paths = healthcheck.write_report(result, out_dir)
        totals = result["totals"]
        _print({
            "files": len(files),
            "rows": totals.get("rows", 0),
            "self_text": totals.get("self_text", 0),
            "self_short": totals.get("self_short", 0),
            "conversations": len(result["conversations"]),
            "by_kind": result["by_kind"],
            "self_by_kind": result["self_by_kind"],
            "length_stats": result["length_stats"],
            "report_md": str(paths["markdown"]),
            "report_json": str(paths["json"]),
        })
        return 0

    if args.cmd == "web":
        if not settings.db_path.exists():
            print(f"数据库不存在：{settings.db_path}，先运行 init", file=sys.stderr)
            return 2
        from humanagent import server
        server.serve(settings, port=args.port, host=args.host, offline=args.offline)
        return 0

    if not settings.db_path.exists():
        print(f"数据库不存在：{settings.db_path}，先运行 init", file=sys.stderr)
        return 2
    conn = db.connect(settings.db_path)

    try:
        if args.cmd == "stats":
            _print({"tables": db.table_counts(conn)})

        elif args.cmd == "ingest":
            target = Path(args.path)
            files = sorted(target.rglob("*.csv")) if target.is_dir() else [target]
            if not files:
                print(f"在 {target} 下没有找到可导入的 csv", file=sys.stderr)
                return 2
            total: dict[str, int] = {"files": len(files)}
            touched: set[int] = set()
            for file in files:
                part = ingest.import_file(conn, settings, file,
                                          self_name=args.self_name or settings.self_name or None,
                                          include_groups=args.include_groups, mask=args.mask,
                                          default_scene=args.scene,
                                          rebuild_context=False)
                touched.update(part.pop("_touched", []))
                for key, value in part.items():
                    if key.startswith("_"):
                        continue
                    total[key] = total.get(key, 0) + value
                print(f"  {file.name}: 读入 {part['read']:,} 行，新插入 {part['inserted']:,}，本人 {part['self_messages']:,} 条", flush=True)
            extra = ingest.finalize_import(conn, settings, conv_ids=touched)
            total.update(extra)
            total["convs_touched"] = len(touched)
            _print(total)

        elif args.cmd == "persona":
            for item in args.set:
                if "=" not in item:
                    print(f"格式应为 键=值：{item}", file=sys.stderr)
                    return 2
                key, value = item.split("=", 1)
                kb.set_persona(conn, key, value)
            _print({"persona": kb.get_persona(conn)})

        elif args.cmd == "fact":
            kb.upsert_facts(conn, [{
                "subject": "我", "predicate": args.predicate,
                "object": args.object_value, "confidence": args.confidence,
            }], evidence="manual")
            _print({"added": 1, "predicate": args.predicate, "object": args.object_value})

        elif args.cmd == "pref":
            kb.upsert_preferences(conn, [{
                "domain": args.domain, "statement": args.statement,
                "polarity": args.polarity, "confidence": 0.95,
            }], evidence="manual")
            _print({"added": 1, "statement": args.statement})

        elif args.cmd == "extract":
            _print(kb.extract_facts(llm, conn, limit=args.limit, workers=args.workers,
                                    max_cost=args.max_cost))

        elif args.cmd == "chat":
            result = agent.reply(
                llm, conn, args.message,
                mode=args.mode,
                partner=args.contact,
                use_tools=False if args.no_tools else None,
            )
            if args.trace:
                trace = (result.get("evidence") or {}).get("tool_trace") or []
                print("trace: " + (json.dumps(trace, ensure_ascii=False) if trace else "[]"),
                      file=sys.stderr)
            _print(result)

        elif args.cmd == "consolidate":
            from humanagent import consolidate
            if args.check:
                _print(consolidate.check_people(conn))
            else:
                names = args.sample if args.sample else (args.name or None)
                model = args.model or (settings.model_reasoner if args.reasoner else "")
                _print(consolidate.consolidate_top(
                    llm, conn, settings,
                    top=args.top,
                    names=names,
                    model=model or None,
                    workers=args.workers,
                    stale_only=args.stale,
                    sample_run=bool(args.sample),
                ))

        elif args.cmd == "people":
            from humanagent import consolidate
            if args.name:
                text = consolidate.read_people(args.name)
                if not text:
                    print(f"没有 {args.name} 的固化文件，先跑 consolidate", file=sys.stderr)
                    return 2
                print(text)
            else:
                _print({"people": consolidate.list_people()})

        elif args.cmd == "cost":
            _print(summarize_calls(settings))

        elif args.cmd == "rescan-scenes":
            contact_scenes = ingest.load_contact_scenes()
            group_scenes = ingest.load_group_scenes()
            updated = 0
            summary: dict[str, int] = {}
            for conv in conn.execute("SELECT id, name, is_group FROM conversations"):
                scene = ingest.match_scene(
                    conv["name"], args.scene, contact_scenes,
                    group_scenes, is_group=bool(conv["is_group"]))
                cur = conn.execute(
                    "UPDATE episodes SET scene=? WHERE conv_id=? AND IFNULL(scene,'') <> ?",
                    (scene, conv["id"], scene))
                updated += cur.rowcount
                summary[scene] = summary.get(scene, 0) + 1
            conn.commit()
            _print({"episodes_updated": updated, "conversations_by_scene": summary,
                    "hint": "场景标签已按 data/contact_scenes.json 重刷"})

    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
