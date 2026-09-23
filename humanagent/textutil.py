"""文本工具：中文检索分词、低信息量判定、脱敏。

没有引入 jieba 等外部依赖，用「CJK 二元切分 + 拉丁词切分」的方式给
SQLite FTS5 提供可索引的 token 串。原因是 FTS5 自带的分词器不会切分中文，
而 trigram 分词器又无法匹配两字查询，二元切分在两者之间最稳。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

CJK_RANGE = "\u4e00-\u9fff\u3400-\u4dbf"
_cjk_run = re.compile(f"[{CJK_RANGE}]+")
_latin_run = re.compile(r"[A-Za-z0-9_]+")
_place_holder = re.compile(r"^\[[^\]]{1,12}\]$")            # [图片] [表情] [语音]
_digit_run = re.compile(r"\d{7,}")

# 纯应答词：这些才是真正的噪声，学不到任何风格
FILLER = {
    "嗯", "嗯嗯", "嗯嗯嗯", "哦", "哦哦", "噢", "好", "好的", "好滴", "好嘞", "行", "行吧",
    "可以", "是的", "对", "对的", "是", "啊", "呀", "哈", "哈哈", "哈哈哈", "哈哈哈哈",
    "呵呵", "嘿", "喂", "在", "在的", "收到", "没事", "谢谢", "多谢", "诶", "哎", "唉",
    "呃", "额", "emmm", "emm", "em", "ok", "okay", "么", "嘛", "哇", "咦", "嗷", "喵", "擦",
}


def _is_noise_char(ch: str) -> bool:
    """标点、空白、表情符号等不承载内容的字符。"""
    if ch.isspace():
        return True
    return unicodedata.category(ch) in {"Po", "Ps", "Pe", "Pd", "Pi", "Pf", "So", "Sk", "Sm"}


def normalize(text: str) -> str:
    """全角转半角、去零宽字符、压缩空白。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> str:
    """生成用于 FTS5 的 token 串（空格分隔）。"""
    text = normalize(text).lower()
    tokens: list[str] = []
    for run in _cjk_run.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    tokens.extend(m.lower() for m in _latin_run.findall(text))
    return " ".join(tokens)


def digest(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def is_low_information(text: str, min_chars: int = 2) -> bool:
    """判断是否为无信息量回复（嗯、好的、哈哈、[表情]……）。

    这类样本在聊天记录里占比极高，不隔离掉会让模型只会敷衍。
    注意：不能只按字数卡。很多人本来就爱发短句，硬性过滤会把他的风格一起滤掉，
    所以这里只剔除「纯应答词」和「去掉标点表情后没有内容」的消息。
    """
    t = normalize(text)
    if not t:
        return True
    stripped = _place_holder.sub("", t)
    stripped = "".join(ch for ch in stripped if not _is_noise_char(ch)).strip().lower()
    if not stripped:
        return True
    if stripped in FILLER:
        return True
    return len(stripped) < min_chars


def mask_sensitive(text: str, replacement: str = "<ID>") -> str:
    """粗粒度脱敏：长数字串（手机号、账号等）。"""
    return _digit_run.sub(replacement, text)


# ---- 敏感信息防线 ----
# 目标：机密类内容既不进知识库，也永远不进模型调用的提示词。
SECRET_KEYWORD = re.compile(
    r"(密码|口令|支付密码|验证码|身份证|银行卡|信用卡|社保|医保|账号|账户|用户名|"
    r"password|passwd|api[_\-\s]?key|secret|token|私钥|username)",
    re.IGNORECASE,
)
ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LONG_DIGITS = re.compile(r"(?<!\d)\d{12,}(?!\d)")

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("身份证", ID_CARD),
    ("手机号", PHONE),
    ("邮箱", EMAIL),
    ("长数字串", LONG_DIGITS),
)


def contains_sensitive(text: str) -> str | None:
    """命中敏感信息就返回类别，否则返回 None。用于拦截入库。"""
    if not text:
        return None
    if SECRET_KEYWORD.search(text):
        return "敏感关键词"
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return label
    return None


def redact(text: str) -> str:
    """把敏感内容替换成占位符。所有进入提示词的文本都要先过这一步。"""
    if not text:
        return text
    out = SECRET_KEYWORD.sub("[敏感]", text)
    # 关键词后面跟的内容一并屏蔽（例如「密码 xxx」「密码：xxx」「password=xxx」）
    out = re.sub(r"\[敏感\][\s:：=＝\-]{0,3}[^\s，。；,;\n]{0,40}", "[敏感已隐藏]", out)
    for label, pattern in SECRET_PATTERNS:
        out = pattern.sub(f"[{label}已隐藏]", out)
    return out


def emoji_stats(text: str) -> tuple[int, int]:
    """返回 (字符数, 表情/符号数量)。"""
    count = 0
    for ch in text:
        if unicodedata.category(ch) in {"So", "Sk"}:
            count += 1
    return len(text), count
