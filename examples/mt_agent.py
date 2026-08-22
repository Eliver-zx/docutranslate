"""单段直译：把 docx 的 JSON 分段协议整个换掉。

名字里的 MT 是历史遗留 —— 这不是"MT 模型专用适配层"，而是小模型跑 docx 的
主路径。分段协议要求模型一边翻译一边维护 id 账本，9B 扛不住：漏键、整块原样
吐回、在字符串值里进复读循环。一段一请求把账本删掉，这些失败模式随之消失。

五种编排在 5 个最难文件（650 段）上的实测：

    编排            缺键/错位   罢工    输出tok
    库模板(id+t)      4.9%     9.4%    68346
    极简(数字键)      0.5%     7.5%    61404
    原文做键         24.6%     6.9%   148405
    只输出值(位置)    10.5%     3.2%    66767
    单段直译          0.0%     3.7%    55844

罢工的 3.7% 全是报关编号 / HS 税则号 / 金额 / 品牌名，本就不该翻，真实失败率 0%。

附一个跨文件跨运行复用的 SQLite 段级缓存：省重复请求，且保证同一原文在整个
语料里得到同一译文。

库源码零改动：唯一耦合点是 enable_mt() 里替换的那一个符号。升级 docutranslate 时
只需确认 docx_translator 仍从 agents.segments_agent 导入 SegmentsTranslateAgent。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
from pathlib import Path

from docutranslate.agents.agent import AgentResultError
from docutranslate.agents.segments_agent import SegmentsTranslateAgent
from docutranslate.utils.json_utils import segments2json_chunks

# Hunyuan-MT 官方模型卡的中↔外模板；对 Qwen 同样是实测最好的写法（越短越稳）
_PROMPT = "把下面的文本翻译成{to_lang}，不要额外解释。\n\n{text}"

_EXPLAIN_PREFIX_RE = re.compile(
    r"^\s*(?:以下是为?翻译|以下是?译文|翻译结果(?:如下)?|翻译如下|译文如下|翻译|译文"
    r"|Translation|Translated\s*text)\s*[:：]\s*",
    re.IGNORECASE,
)

# ponytail: 模块级计数器，运行器是单事件循环单进程，无需加锁
stats = {"cache_hit": 0, "request": 0, "echo": 0, "failed": 0}

# 回声长度阈值：实测最长的合法回声是 34 字符（"S ARN 09 11 4 DK AAR A013 29.11.04"），
# 40 字符以内一律当作"本就不该翻的代码"，可安全缓存。
ECHO_CACHE_MAX = 40

# 跨文件共享的全局并发闸。库自带的 concurrent 是"每文件最多几个在飞"，
# 挡不住 N 个文件同时各开 N 个。两级叠加：全局 20（喂满服务端 --max-num-seqs）
# × 每文件 3（单个文件不独占队列）。要填满 20 需要至少 7 个文件在飞，
# 所以 file_concurrent 要相应调高。
_global_limit: int | None = None
_global_sem: asyncio.Semaphore | None = None


def set_global_concurrency(n: int | None) -> None:
    """在事件循环外只记数值；Semaphore 首次用到时才建，避免绑错循环。"""
    global _global_limit, _global_sem
    _global_limit, _global_sem = n, None


def sanitize(result: str) -> str:
    """剥掉 MT 模型常见的回声与解释外壳，只留译文。"""
    out = result.split("</input>")[0].replace("<input>", "").strip()
    while True:
        stripped = _EXPLAIN_PREFIX_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped.strip()
    if out.startswith("```") and out.endswith("```"):
        lines = out.splitlines()
        out = "\n".join(lines[1:-1]).strip()
    return out


def _norm(s: str) -> str:
    return " ".join(s.split())


def is_untranslated_echo(source: str, translated: str) -> bool:
    """译文与原文一致、长度足够、且含非 CJK 字母 —— 判定为模型没翻，原样吐回。

    短串（BTI、2024 这类代号）允许原样保留，不算回声。
    """
    src, dst = source.strip(), translated.strip()
    if len(src) <= 5:
        return False
    if _norm(src) != _norm(dst):
        return False
    return any(ch.isalpha() and not ("\u4e00" <= ch <= "\u9fff") for ch in src)


class MTCache:
    """段级译文缓存：键为 sha256(目标语言 \\0 模型 \\0 原文)。

    同模板的批量文档（证书、表单）重复段极多，命中率直接决定实际请求量。
    自定义提示词与术语表不进键 —— 改动这两者需自行删除缓存文件。
    """

    def __init__(self, path: str | Path, to_lang: str, model_id: str) -> None:
        self.path = str(path)
        self.to_lang = to_lang
        self.model_id = model_id or ""
        self._mem: dict[str, str] = {}
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # 长连接：`with sqlite3.connect(...)` 只提交事务、不关连接，每次 get/put
        # 新建会泄漏 fd（全语料约 74k 次）。WAL + busy_timeout 让并发写不炸。
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS t (h TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def _hash(self, text: str) -> str:
        raw = f"{self.to_lang}\x00{self.model_id}\x00{text}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, text: str) -> str | None:
        h = self._hash(text)
        if h in self._mem:
            return self._mem[h]
        row = self._conn.execute("SELECT v FROM t WHERE h=?", (h,)).fetchone()
        if row is None:
            return None
        self._mem[h] = row[0]
        return row[0]

    def put(self, text: str, translated: str) -> None:
        h = self._hash(text)
        self._mem[h] = translated
        self._conn.execute(
            "INSERT OR REPLACE INTO t (h, v) VALUES (?, ?)", (h, translated)
        )
        self._conn.commit()


def _result_handler(result: str, prompt: str, logger) -> str:
    out = sanitize(result)
    if not out:
        raise AgentResultError("MT 单段翻译结果为空")
    return out


def _error_result_handler(prompt: str, logger) -> None:
    """重试耗尽：返回 None，由调用方保留原文并计入失败。"""
    return None


def _rebuild(
    values: list[str], merged_indices_list: list[tuple[int, int]]
) -> list[str]:
    """把 segments2json_chunks 拆开的超长段重新拼回去。"""
    out: list[str] = []
    last = 0
    for start, end in merged_indices_list:
        out.extend(values[last:start])
        out.append("".join(map(str, values[start:end])))
        last = end
    out.extend(values[last:])
    return out


class MTSegmentsAgent(SegmentsTranslateAgent):
    """一段一次请求，按原文去重，命中缓存的段不发请求。"""

    def __init__(self, config, cache_path: str | Path | None = None) -> None:
        super().__init__(config)
        self.cache = (
            MTCache(cache_path, config.to_lang, config.model_id) if cache_path else None
        )

    def send_segments(self, segments: list[str], chunk_size: int) -> list[str]:
        raise RuntimeError("MT 模式仅支持异步路径，请调用 workflow.translate_async()")

    async def send_async(self, *args, **kwargs):
        """每个请求先过全局闸，再走库自带的每文件信号量。"""
        global _global_sem
        if _global_limit is None:
            return await super().send_async(*args, **kwargs)
        if _global_sem is None:
            _global_sem = asyncio.Semaphore(_global_limit)
        async with _global_sem:
            return await super().send_async(*args, **kwargs)

    async def send_segments_async(
        self, segments: list[str], chunk_size: int
    ) -> list[str]:
        indexed_originals, _chunks, merged = await asyncio.to_thread(
            segments2json_chunks, segments, chunk_size
        )
        translated = dict(indexed_originals)
        pending: dict[str, list[str]] = {}
        hits = 0

        for key, text in indexed_originals.items():
            if not text.strip():
                continue
            cached = self.cache.get(text) if self.cache else None
            if cached is not None:
                translated[key] = cached
                hits += 1
            else:
                pending.setdefault(text, []).append(key)

        texts = list(pending)
        stats["cache_hit"] += hits
        stats["request"] += len(texts)
        self.logger.info(
            f"MT 单段模式: 共 {len(indexed_originals)} 段, 缓存命中 {hits} 段, "
            f"待请求 {len(texts)} 个唯一文本"
        )

        if texts:
            prompts = [_PROMPT.format(to_lang=self.to_lang, text=t) for t in texts]
            answers = await self.send_prompts_async(
                prompts=prompts,
                pre_send_handler=self._pre_send_handler,
                result_handler=_result_handler,
                error_result_handler=_error_result_handler,
            )
            for text, answer in zip(texts, answers):
                keys = pending[text]
                if answer is None:
                    stats["failed"] += len(keys)
                    continue  # 保留原文
                if is_untranslated_echo(text, answer):
                    # 温度为 0 时重试等价于原样重发，不重试。
                    # 实测 240 个回声全部 ≤40 字符，全是报关编号 / HS 税则号 / 金额 /
                    # 品牌名（"3707-9034"、"E DET 25 08 3 DE HAM W029"），本就不该翻，
                    # 原样返回是正确行为 —— 这类进缓存，省掉全语料里的成千次重发。
                    # 超过阈值的回声才是可疑的真失败，采用输出但不写缓存。
                    stats["echo"] += len(keys)
                    if self.cache and len(text.strip()) <= ECHO_CACHE_MAX:
                        self.cache.put(text, answer)
                    elif self.cache:
                        self.logger.warning(
                            f"长文本原样返回，疑似真失败，不写缓存: {text[:60]!r}"
                        )
                elif self.cache:
                    self.cache.put(text, answer)
                for key in keys:
                    translated[key] = answer

        return _rebuild(list(translated.values()), merged)


def enable_mt(cache_path: str | Path | None) -> None:
    """把 docx 链路的分段 Agent 换成 MT 单段实现。"""
    import docutranslate.translator.ai_translator.docx_translator as docx_translator

    docx_translator.SegmentsTranslateAgent = lambda config: MTSegmentsAgent(
        config, cache_path
    )


def demo() -> None:
    import tempfile

    assert sanitize("译文：你好") == "你好"
    assert sanitize("翻译结果如下：\n你好") == "你好"
    assert sanitize("Translation: hello") == "hello"
    assert sanitize("```\n你好\n```") == "你好"
    assert sanitize("你好</input>残留") == "你好"
    assert sanitize("  你好  ") == "你好"

    assert is_untranslated_echo("Country of issue", "Country of issue")
    assert is_untranslated_echo("Country of issue", "  Country   of issue ")
    assert not is_untranslated_echo("Country of issue", "签发国家")
    assert not is_untranslated_echo("BTI", "BTI")  # 短代号原样保留不算回声
    assert not is_untranslated_echo("这是一段中文文本", "这是一段中文文本")  # 无外文

    # 阈值必须容得下实测最长的合法回声
    assert len("S ARN 09 11 4 DK AAR A013 29.11.04") <= ECHO_CACHE_MAX
    assert len("E DET 25 08 3 DE HAM W029") <= ECHO_CACHE_MAX

    assert _rebuild(["a", "b", "c", "d"], [(1, 3)]) == ["a", "bc", "d"]
    assert _rebuild(["a", "b"], []) == ["a", "b"]

    with tempfile.TemporaryDirectory() as tmp:
        cache = MTCache(Path(tmp) / "sub" / "c.sqlite3", "简体中文", "m")
        assert cache.get("hello") is None
        cache.put("hello", "你好")
        assert cache.get("hello") == "你好"
        assert (
            MTCache(Path(tmp) / "sub" / "c.sqlite3", "简体中文", "m").get("hello")
            == "你好"
        )
        # 换模型即换键，不复用
        assert (
            MTCache(Path(tmp) / "sub" / "c.sqlite3", "简体中文", "other").get("hello")
            is None
        )

        # 连接不能每次新建：1000 次读写后仍只占一个 fd
        import os

        proc = Path(f"/proc/{os.getpid()}/fd")
        before = len(list(proc.iterdir())) if proc.is_dir() else None
        for i in range(1000):
            cache.put(f"k{i}", f"v{i}")
            assert cache.get(f"k{i}") == f"v{i}"
        if before is not None:  # Linux 才有 /proc，macOS 上跳过
            assert len(list(proc.iterdir())) <= before + 2, "疑似连接泄漏"

    # 全局闸：未设置时不拦，设置后按数值建一次
    assert _global_limit is None and _global_sem is None
    set_global_concurrency(20)
    assert _global_limit == 20 and _global_sem is None  # 尚未进事件循环，不该提前建

    async def _touch():
        agent = MTSegmentsAgent.__new__(MTSegmentsAgent)
        seen = []

        async def fake(*a, **k):
            seen.append(len(a))
            return "ok"

        agent_cls = type(agent)
        original = SegmentsTranslateAgent.send_async
        SegmentsTranslateAgent.send_async = fake
        try:
            assert await agent_cls.send_async(agent, "x") == "ok"
        finally:
            SegmentsTranslateAgent.send_async = original
        assert _global_sem is not None, "首次调用应建出 Semaphore"

    asyncio.run(_touch())
    set_global_concurrency(None)

    print("mt_agent 自检通过")


if __name__ == "__main__":
    demo()
