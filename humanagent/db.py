"""SQLite 知识库。

单文件数据库，方便备份与随论文提交。记忆分三张表：
    episodes        情节记忆：一串带上下文的真实对话片段
    facts           语义记忆：关于你的事实（随时间更新，旧的置为 superseded）
    preferences     偏好画像：你在各领域喜欢 / 排斥什么
    feedback        反馈闭环：草稿、你的选择、你的最终文本

「说话方式」不落库，写在 data/style_profile.json 里（三档 + 真实原话样例）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE,              -- 联系人 / 群名
    is_group    INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    conv_id     INTEGER REFERENCES conversations(id),
    ts          TEXT,                     -- 原始时间戳
    ts_epoch    INTEGER,                  -- 便于按时间衰减排序
    role        TEXT,                     -- self / other
    text        TEXT NOT NULL,
    context     TEXT,                     -- 该条之前的若干条对话，拼成文本
    scene       TEXT,                     -- 场景标签，可后补
    is_short    INTEGER DEFAULT 0,        -- 低信息量标记
    is_holdout  INTEGER DEFAULT 0,        -- 是否划入评测留出集
    used_in_sft INTEGER DEFAULT 0,        -- 是否进入过训练集
    content_hash TEXT UNIQUE,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY,
    subject     TEXT NOT NULL,            -- 一般固定为「我」
    predicate   TEXT NOT NULL,            -- 例如 就读院校 / 偏好 / 正在做
    object      TEXT NOT NULL,
    confidence  REAL DEFAULT 0.7,
    evidence    TEXT,                     -- 来源 episode id，逗号分隔
    status      TEXT DEFAULT 'active',    -- active / superseded
    valid_from  TEXT,
    valid_to    TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject, predicate, object)
);

CREATE TABLE IF NOT EXISTS preferences (
    id          INTEGER PRIMARY KEY,
    domain      TEXT,                     -- 学习 / 饮食 / 社交 / 娱乐 …
    statement   TEXT NOT NULL,            -- 我喜欢在晚上写代码
    polarity    TEXT,                     -- like / dislike / neutral
    confidence  REAL DEFAULT 0.7,
    evidence    TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    status      TEXT DEFAULT 'active',    -- active / merged
    UNIQUE(domain, statement)
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY,
    ts          TEXT DEFAULT CURRENT_TIMESTAMP,
    situation   TEXT NOT NULL,            -- 当时的对话上下文
    candidates  TEXT NOT NULL,            -- JSON 数组，模型给的三条候选
    chosen_idx  INTEGER,                  -- 你选中的序号，-1 表示全弃用
    final_text  TEXT,                     -- 你实际发出去的文本
    edit_ratio  REAL,                     -- 你改了多大比例
    note        TEXT
);

-- 抽取进度表：记录哪些情节已经做过事实抽取，避免重复调模型花冤枉钱
CREATE TABLE IF NOT EXISTS extract_log (
    episode_id  INTEGER PRIMARY KEY,
    ts          TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 人设卡：你自己给的身份与风格设定，直接进入每次生成的提示词
CREATE TABLE IF NOT EXISTS persona (
    key         TEXT PRIMARY KEY,   -- 姓名 / 昵称 / 身份 / 城市 / 风格 / 雷区 / 称呼习惯 …
    value       TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 索引：token 列存二元切分结果，原文另存
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    tokens, episode_id UNINDEXED, tokenize='unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    tokens, fact_id UNINDEXED, tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_episodes_conv ON episodes(conv_id, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_episodes_holdout ON episodes(is_holdout);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status, predicate);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> Path:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()
    return Path(db_path)


def _migrate(conn: sqlite3.Connection) -> None:
    """给老库补上后加的字段。"""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(preferences)")}
    if "status" not in columns:
        conn.execute("ALTER TABLE preferences ADD COLUMN status TEXT DEFAULT 'active'")


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = ["conversations", "episodes", "facts", "preferences", "feedback", "persona"]
    out: dict[str, int] = {}
    for name in names:
        out[name] = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    out["episodes_trainable"] = conn.execute(
        """SELECT COUNT(*) FROM episodes e JOIN conversations c ON c.id = e.conv_id
           WHERE e.role='self' AND e.is_short=0 AND e.is_holdout=0 AND c.is_group = 0"""
    ).fetchone()[0]
    out["episodes_group"] = conn.execute(
        """SELECT COUNT(*) FROM episodes e JOIN conversations c ON c.id = e.conv_id
           WHERE c.is_group = 1"""
    ).fetchone()[0]
    return out
