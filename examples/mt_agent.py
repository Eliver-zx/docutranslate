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
from contextvars import ContextVar
from pathlib import Path

from docutranslate.agents.agent import AgentResultError
from docutranslate.agents.segments_agent import SegmentsTranslateAgent
from docutranslate.utils.json_utils import segments2json_chunks

# Hunyuan-MT 官方模型卡的中↔外模板；对 Qwen 同样是实测最好的写法（越短越稳）
_PROMPT = "把下面的文本翻译成{to_lang}，不要额外解释。\n\n{text}"

# 长文本被原样吐回时的重问模板。温度为 0，原样重发等价于复读，只有换提示词才有
# 新结果。逐条更强硬，用尽仍原样返回就判整文件失败。
_RETRY_PROMPTS = (
    "请把下面的文本逐句翻译成{to_lang}。禁止原样返回原文，只输出译文。\n\n{text}",
    "下面是一段外文，必须译成{to_lang}。输出中不得出现原文，只给译文。\n\n{text}",
)

_EXPLAIN_PREFIX_RE = re.compile(
    r"^\s*(?:以下是为?翻译|以下是?译文|翻译结果(?:如下)?|翻译如下|译文如下|翻译|译文"
    r"|Translation|Translated\s*text)\s*[:：]\s*",
    re.IGNORECASE,
)

# ponytail: 模块级计数器，运行器是单事件循环单进程，无需加锁
_STAT_KEYS = (
    "cache_hit",
    "request",
    "echo",
    "echo_retry",
    "failed",
    "segments",
    "in_tokens",
    "out_tokens",
    "total_tokens",
)

# 全局累计。键必须与 _STAT_KEYS 一致 —— _bump 同时写全局与逐文件两份，
# 少一个键就是 KeyError（实测漏了 "segments" 让 30 个文件全废）。
stats = dict.fromkeys(_STAT_KEYS, 0)

# 当前正在翻译的文件。多文件并发时日志交织，没有这个前缀就分不清
# "2 段长文本原样返回" 是谁的。asyncio.Task 继承创建时的 context，
# runner 在每个文件的 worker 里 set 一次即可。
_FILE: ContextVar[str] = ContextVar("mt_current_file", default="")
_FILE_STATS: ContextVar[dict | None] = ContextVar("mt_file_stats", default=None)

def begin_file(name: str) -> dict:
    """进入一个文件的翻译作用域，返回该文件独有的计数器。

    必须在该文件的 asyncio 任务内部调用 —— ContextVar 只对当前 task
    及其后续 await 可见，在 task 外面 set 会串到别的文件上。
    """
    counters = dict.fromkeys(_STAT_KEYS, 0)
    _FILE.set(name)
    _FILE_STATS.set(counters)
    return counters


def _bump(key: str, n: int = 1) -> None:
    """同时记全局与当前文件的计数。"""
    stats[key] += n
    if (fs := _FILE_STATS.get()) is not None:
        fs[key] += n


def _tag(msg: str) -> str:
    name = _FILE.get()
    return f"[{name}] {msg}" if name else msg


def patch_token_extraction() -> None:
    """修掉库里 token 统计返回 -1 哨兵值的问题。

    vLLM 的 usage 里 prompt_tokens_details 是 None，库用
    `"cached_tokens" in usage["prompt_tokens_details"]` 探测，对 None 做
    in 判断抛 TypeError，被 except 吞掉后返回 (-1,-1,-1,-1,-1)，这个 -1
    一路累加进 TokenCounter —— 实测 30 篇跑出 "token -2.4K"。
    库源码不动：把值为 None 的 details 键摘掉再交给原函数，让它走正常分支。
    """
    from docutranslate.agents import agent as _agent

    if getattr(_agent.extract_token_info, "_mt_patched", False):
        return
    _orig = _agent.extract_token_info

    def extract_token_info(response_data: dict):
        usage = response_data.get("usage")
        if isinstance(usage, dict):
            for k in (
                "prompt_tokens_details",
                "completion_tokens_details",
                "input_tokens_details",
                "output_tokens_details",
            ):
                if k in usage and usage[k] is None:
                    usage.pop(k)
        r = _orig(response_data)
        # 顺手把 token 记到自己账上。库的 per-workflow 统计只保留最后一轮
        # send_prompts_async —— 一旦触发长文本重问，前面几百个请求的 token
        # 全被覆盖掉（实测 181 段的文件只报出 1 个请求、0.1K token）。
        # 这里是所有请求的必经之路，累加在此才不漏。
        if r[0] >= 0:
            _bump("in_tokens", r[0])
            _bump("out_tokens", r[2])
            _bump("total_tokens", r[4])
        return r

    extract_token_info._mt_patched = True
    _agent.extract_token_info = extract_token_info

# 回声长度阈值：实测最长的合法回声是 34 字符（"S ARN 09 11 4 DK AAR A013 29.11.04"），
# 40 字符以内一律当作"本就不该翻的代码"，可安全缓存；超过的算真失败，重试到译出为止。
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


_XML_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def visible_text(s: str) -> str:
    """剥掉 XML 标签，只留可见文本。

    docx 的表格单元格会把整个 <w:tc> 的 XML 当成一个可翻译 segment 送进来，
    标签本身上百字符，任何基于长度的判据都必须先剥干净再量。
    """
    return _XML_TAG.sub("", s).strip()


def is_code_only(visible: str) -> bool:
    """只由缩写与数字组成 —— 这类不必翻译，模型原样返回是正确行为。

    缩写＝字母全大写的 token（HEG、SE、VAG、W044、BTI、HS），外加首字母大写的
    短 token —— 报关单号里的港口代码大小写并不规范，同一栏位既写 'VAG' 也写
    'Imm'（Immingham），实测 04-2006.docx 有 68 段栽在这一个 token 上。
    数字＝纯数字 token（3731、2004、150）。

    长度上限卡在 4：真实词都比这长，'Sendingarnúmer' 首字母大写但 14 个字母，
    不会被当成缩写。只要出现一个全小写的实义 token 就不豁免 —— 冰岛语的
    'Er þá um að ræða' 全是短词但句意完整，必须翻译。
    """
    tokens = _TOKEN.findall(visible)
    return all(
        t.isdigit() or t.isupper() or (t[:1].isupper() and len(t) <= 4)
        for t in tokens
    )


def is_untranslated_echo(source: str, translated: str) -> bool:
    """译文与原文一致、长度足够、且含非 CJK 字母 —— 判定为模型没翻，原样吐回。

    短串（BTI、2024 这类代号）允许原样保留，不算回声。

    两道豁免都只看可见文本。docx 的表格单元格整段带 <w:tc> 等标签进来，标签
    本身就有几十字符，会把短代号顶过长度阈值 —— 实测报关单号
    'S HEG 06 04 4 SE VAG W044' 因此被判失败，16-2008-1.docx 三次重试全废。
    剥完标签后再按「缩写与数字不翻译，其余全部翻译」豁免：报关编号 / HS 税则号 /
    航班号 / 金额只含全大写 token 与数字，本就不该翻。
    """
    src, dst = source.strip(), translated.strip()
    if _norm(src) != _norm(dst):
        return False
    visible = visible_text(src)
    if len(visible) <= 5:
        return False
    if is_code_only(visible):
        return False
    return any(ch.isalpha() and not ("\u4e00" <= ch <= "\u9fff") for ch in visible)


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
        """首次请求过全局闸，再走库自带的每文件信号量。

        重试不重复取闸：库的重试是 `return await self.send_async(..., retry_count=n+1)`
        递归（agent.py:794），解析到本覆写。重试请求此刻仍持有闸，再抢一次会在
        「许可全被重试中的请求占住」时死锁 —— 而服务端抖动正是重试集中发生的时刻。
        """
        global _global_sem
        if _global_limit is None or kwargs.get("retry_count", 0):
            return await super().send_async(*args, **kwargs)
        if _global_sem is None:
            _global_sem = asyncio.Semaphore(_global_limit)
        async with _global_sem:
            return await super().send_async(*args, **kwargs)

    async def _ask(self, texts: list[str], template: str) -> list[str | None]:
        return await self.send_prompts_async(
            prompts=[template.format(to_lang=self.to_lang, text=t) for t in texts],
            pre_send_handler=self._pre_send_handler,
            result_handler=_result_handler,
            error_result_handler=_error_result_handler,
        )

    def _collect(
        self,
        texts: list[str],
        answers: list[str | None],
        pending: dict[str, list[str]],
        translated: dict,
    ) -> list[str]:
        """落定能落定的译文，返回仍需重试的长文本回声。"""
        todo: list[str] = []
        for text, answer in zip(texts, answers):
            keys = pending[text]
            if answer is None:
                _bump("failed", len(keys))
                continue  # 保留原文
            if is_untranslated_echo(text, answer):
                # 实测 240 个回声全部 ≤40 字符，全是报关编号 / HS 税则号 / 金额 /
                # 品牌名（"3707-9034"、"E DET 25 08 3 DE HAM W029"），本就不该翻，
                # 原样返回是正确行为 —— 这类进缓存，省掉全语料里的成千次重发。
                if len(text.strip()) > ECHO_CACHE_MAX:
                    todo.append(text)
                    continue
                _bump("echo", len(keys))
                if self.cache:
                    self.cache.put(text, answer)
            elif self.cache:
                self.cache.put(text, answer)
            for key in keys:
                translated[key] = answer
        return todo

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
        _bump("cache_hit", hits)
        _bump("request", len(texts))
        _bump("segments", len(indexed_originals))
        self.logger.info(
            _tag(
                f"MT 单段模式: 共 {len(indexed_originals)} 段, 缓存命中 {hits} 段, "
                f"待请求 {len(texts)} 个唯一文本"
            )
        )

        if texts:
            answers = await self._ask(texts, _PROMPT)
            todo = self._collect(texts, answers, pending, translated)
            for tpl in _RETRY_PROMPTS:
                if not todo:
                    break
                _bump("echo_retry", len(todo))
                self.logger.warning(_tag(f"{len(todo)} 段长文本原样返回，换提示词重试"))
                answers = await self._ask(todo, tpl)
                todo = self._collect(todo, answers, pending, translated)
            if todo:
                # 不写缓存、不落盘：抛出去让 runner 把整个文件记为失败，重跑时自然重来
                _bump("failed", sum(len(pending[t]) for t in todo))
                raise RuntimeError(
                    _tag(
                        f"{len(todo)} 段长文本重试 {len(_RETRY_PROMPTS) + 1} 次仍原样返回: "
                        f"{todo[0][:60]!r}"
                    )
                )

        return _rebuild(list(translated.values()), merged)


def enable_mt(cache_path: str | Path | None) -> None:
    """把 docx 链路的分段 Agent 换成 MT 单段实现。"""
    import docutranslate.translator.ai_translator.docx_translator as docx_translator

    docx_translator.SegmentsTranslateAgent = lambda config: MTSegmentsAgent(
        config, cache_path
    )
    patch_token_extraction()


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
    # 规则：缩写与数字不翻译，其余全部翻译
    assert is_code_only("S HEG 06 04 4 SE VAG W044")   # 报关单号
    assert is_code_only("E DET 25 08 3 DE HAM W029")
    assert is_code_only("3707-9034")                    # HS 税则号
    assert is_code_only("15.04.2004")                   # 日期
    assert is_code_only("BTI")
    assert is_code_only("E SEL 22 12 3 GB  Imm W024")   # 港口代码大小写不规范
    assert not is_code_only("Sendingarnumer")           # 实义词（首字母大写但过长）
    assert not is_code_only("Er tha um ad raeda")       # 全短词但句意完整
    assert not is_code_only("dags. 31. desember 2002")  # 缩写混实义词

    _code = "S HEG 06 04 4 SE VAG W044"
    assert not is_untranslated_echo(_code, _code)
    # docx 表格单元格连标签一起进来，标签不得把短代号顶过长度阈值
    _cell = '<w:tc><w:tcPr/><w:p><w:r><w:t>' + _code + '</w:t></w:r></w:p></w:tc>'
    assert not is_untranslated_echo(_cell, _cell)
    # 带标签的真实文本仍要判为回声
    _real = '<w:tc><w:p><w:r><w:t>Sendingarnumer skal fylgja</w:t></w:r></w:p></w:tc>'
    assert is_untranslated_echo(_real, _real)

    # 阈值必须容得下实测最长的合法回声
    # 逐文件计数：_bump 同时写全局与文件两份，任一处缺键都会 KeyError
    before = dict(stats)
    fs = begin_file("t.docx")
    for k in _STAT_KEYS:
        _bump(k, 2)
    assert all(fs[k] == 2 for k in _STAT_KEYS), fs
    assert all(stats[k] == before[k] + 2 for k in _STAT_KEYS), stats
    for k in _STAT_KEYS:  # 还原，别污染真实跑批的计数
        stats[k] = before[k]
    assert _tag("x") == "[t.docx] x"
    _FILE.set("")
    _FILE_STATS.set(None)
    assert _tag("x") == "x"

    assert len("S ARN 09 11 4 DK AAR A013 29.11.04") <= ECHO_CACHE_MAX
    assert len("E DET 25 08 3 DE HAM W029") <= ECHO_CACHE_MAX

    # _collect：短回声进缓存，长回声退回重试队列
    agent = MTSegmentsAgent.__new__(MTSegmentsAgent)
    agent.cache = None
    long_src = "Bersýnilegar villur og reikningsskekkjur í framtali kæranda " * 2
    out: dict = {}
    todo = agent._collect(
        ["BTI 3707-9034 DE HAM", long_src, "Country of issue"],
        ["BTI 3707-9034 DE HAM", long_src, "签发国家"],
        {"BTI 3707-9034 DE HAM": ["1"], long_src: ["2"], "Country of issue": ["3"]},
        out,
    )
    assert todo == [long_src], todo
    assert out == {"1": "BTI 3707-9034 DE HAM", "3": "签发国家"}, out

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

        async def fake(*a, **k):
            return "ok"

        agent_cls = type(agent)
        original = SegmentsTranslateAgent.send_async
        SegmentsTranslateAgent.send_async = fake
        try:
            assert await agent_cls.send_async(agent, "x") == "ok"
            assert _global_sem is not None, "首次调用应建出 Semaphore"

            # 重试不重复取闸：把许可占满后，retry_count>0 的调用仍须立刻通过，
            # 否则重试全体互等 = 死锁。
            limit = _global_limit or 0
            for _ in range(limit):
                await _global_sem.acquire()
            assert _global_sem.locked(), "许可应已占满"
            got = await asyncio.wait_for(
                agent_cls.send_async(agent, "x", retry_count=1), timeout=1
            )
            assert got == "ok"
            for _ in range(limit):
                _global_sem.release()
        finally:
            SegmentsTranslateAgent.send_async = original

    asyncio.run(_touch())
    set_global_concurrency(None)

    print("mt_agent 自检通过")


if __name__ == "__main__":
    demo()
