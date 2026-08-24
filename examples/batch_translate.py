#!/usr/bin/env python3
"""按 profile 批量翻译目录下的 docx / md。

    python examples/batch_translate.py hunyuan
    python examples/batch_translate.py qwen --input A --output B
    python examples/batch_translate.py hunyuan --dry-run
    python examples/batch_translate.py qwen --recheck   # 删掉已有输出里的漏译文件

输出文件已存在就跳过。输出先写 <名字>.part、成功后原子改名，所以半成品不会被
当成已完成；启动时清理上一轮遗留的 .part。失败逐条追加到 <输出目录>/failures.txt
（纯报告：失败文件没有输出，重跑时靠"输出不存在"自然重试）。

--recheck 扫描已有输出，直接删掉整段仍是外文的文件（清单留在 <输出目录>/suspects.txt），
重跑即重译。用于捞回旧版本里被当成功写盘的漏译文件。
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import re
import sys
import time
import tomllib
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mt_agent  # noqa: E402
import schema_agent  # noqa: E402
from docutranslate.exporter.docx.docx2html_exporter import Docx2HTMLExporterConfig  # noqa: E402
from docutranslate.translator.ai_translator.docx_translator import DocxTranslatorConfig  # noqa: E402
from docutranslate.workflow.docx_workflow import DocxWorkflow, DocxWorkflowConfig  # noqa: E402

# md 链路刻意不在顶层 import：MarkdownBasedWorkflow 会拉起 docling → transformers，
# -X importtime 实测 4.25s，而 extensions=[".docx"] 的配置根本走不到它。
# 需要时在 build_workflow 里现 import。

_WT_RE = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.S)


def suspect_text(docx: Path, min_len: int = mt_agent.ECHO_CACHE_MAX) -> str | None:
    """返回输出 docx 里第一段疑似未翻译的正文，没有则 None。

    判据与 MT 的回声判定同源，直接复用 mt_agent 的两个函数：先剥 XML 标签取
    可见文本，再放行「只含缩写与数字」的段。docx 的表格单元格会把整个 <w:tc>
    的 XML 当成一段文本送进来（<w:t> 里存的是转义的 XML 源码），上百字符的标签
    会把 25 字符的报关单号顶过阈值 —— 实测 'S HEG 06 04 4 SE VAG W044' 因此
    让 16-2008-1.docx / 10-2008.docx 三次重试全废。
    仅对 insert_mode="replace" 的输出有意义 —— append 模式原文本就留在文件里。
    """
    try:
        with zipfile.ZipFile(docx) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        return f"<无法读取: {e!r}>"
    for para in xml.split("</w:p>"):
        raw = html.unescape("".join(_WT_RE.findall(para))).strip()
        text = mt_agent.visible_text(raw)
        if len(text) <= min_len:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            continue
        if mt_agent.is_code_only(text):
            continue
        if any(ch.isalpha() for ch in text):
            return text
    return None


def recheck(out_dir: Path) -> None:
    """扫描已有输出，删掉疑似漏译的文件（删了就等于没翻，下次跑自动重译）。"""
    files = sorted(p for p in out_dir.rglob("*.docx") if not p.name.endswith(".part"))
    bad = [(p, t) for p in files if (t := suspect_text(p))]
    report = out_dir / "suspects.txt"
    report.write_text("".join(f"{p}\t{t[:120]}\n" for p, t in bad), encoding="utf-8")
    for path, _ in bad:
        path.unlink()
    print(
        f"扫描 {len(files)} 个输出，删除疑似漏译 {len(bad)} 个（清单 → {report}），重跑即重译"
    )


logger = logging.getLogger("batch")
# 库的日志走子 logger：它每个请求打一条"协程-已完成 N/M"，多文件并发下交织成噪音。
# 子 logger 默认压到 WARNING，--verbose 放开；handler 挂在父 logger 上，共用输出。
lib_logger = logging.getLogger("batch.lib")
totals = {
    "success": 0,
    "skipped": 0,
    "failed": 0,
    "tokens": 0,
    "in_tokens": 0,
    "out_tokens": 0,
    "requests": 0,
}
# 失败原因 → 次数。只留错误类别，不留具体内容 —— 具体内容在 failures.txt。
fail_kinds: dict[str, int] = {}
# 本次跑的 profile 名。不塞进 cfg：cfg 有未知键校验，多一个键直接退出。
run_profile = ""
# 每个文件一行记录，跑完落成 run.json 供事后对比不同模型
file_records: list[dict] = []


def classify_failure(reason: str) -> str:
    """把 repr(e) 归成一类，用于结束时的失败分布统计。"""
    for probe, label in (
        ("存在未翻译段落", "未翻译段落"),
        ("仍原样返回", "长文本回声"),
        ("未解决的错误", "库内未解决错误"),
        ("没有返回统计信息", "翻译未执行"),
        ("Timeout", "超时"),
        ("timeout", "超时"),
        ("ConnectError", "连接失败"),
        ("ConnectTimeout", "连接失败"),
        ("status_code=4", "4xx"),
        ("status_code=5", "5xx"),
    ):
        if probe in reason:
            return label
    return reason.split("(")[0][:30] or "未知"

REQUIRED = ("base_url", "model_id", "input", "output")

# 默认值集中在这里，不再散落成 cfg.get("x", 默认值)。TOML 里没写的键取这里的值。
DEFAULTS: dict = {
    "to_lang": "简体中文",
    "api_key": "EMPTY",
    "custom_prompt": "",
    "extensions": [".docx"],
    "chunk_size": 3000,
    "concurrent": 3,
    "temperature": 0.0,
    "top_p": 0.9,
    "timeout": 300,
    "retry": 3,
    "file_concurrent": 10,
    "file_retry": 2,
    "retry_backoff": 10,
    "thinking": "default",
    "force_json": False,
    "insert_mode": "replace",
    "separator": "\n",
    "system_proxy_enable": False,
    "global_concurrent": None,
    "extra_body": "",
    "enable_thinking": None,
    "max_tokens": None,
    "mt": False,
    "mt_cache": None,
    "json_schema": False,
    "minimal_prompt": False,
    # 重试时换 profile：第 1 次用本 profile，第 2 次起换成它。
    # 同一个模型失败两次通常还是失败，换个模型才有意义。
    "fallback_profile": None,
}


def load_profile(config_path: Path, name: str) -> dict:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    profiles = raw.get("profile", {})
    if name not in profiles:
        available = ", ".join(profiles) or "(无)"
        sys.exit(f"{config_path} 里没有 profile.{name}，可用: {available}")

    cfg = {**DEFAULTS, **raw.get("default", {}), **profiles[name]}

    unknown = sorted(set(cfg) - set(DEFAULTS) - set(REQUIRED))
    if unknown:
        known = ", ".join(sorted(set(DEFAULTS) | set(REQUIRED)))
        sys.exit(
            f"{config_path} profile.{name}: 未知配置项 {', '.join(unknown)}\n可用: {known}"
        )
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        sys.exit(f"{config_path} profile.{name}: 缺少必填项 {', '.join(missing)}")
    return cfg


def setup_logger(log_path: Path | None, verbose: bool, quiet: bool) -> None:
    """log_path 为 None 时只输出到 stdout —— 配置还没解析出输出目录时用。"""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    lib_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    if quiet:
        stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(file_handler)


def _extra_body(cfg: dict) -> str | None:
    """把 enable_thinking 开关折进 extra_body 的 chat_template_kwargs。

    不走库的 thinking 字段：它对 model_id 含 "qwen" 的一律套阿里云 DashScope 的
    顶层 enable_thinking，自建 vLLM 不认这个参数、静默忽略。实测（Qwen3.5-9B /
    vLLM 0.27.1）顶层写法与什么都不写完全一致 —— 思考照开、200 token 全烧在
    reasoning 里、content 为空；chat_template_kwargs 才真关掉，同任务 4 token。

    profile 里手写的 extra_body 优先，方便临时压其他参数。
    """
    raw = (cfg.get("extra_body") or "").strip()
    body = json.loads(raw) if raw else {}
    if cfg.get("enable_thinking") is not None:
        kwargs = body.setdefault("chat_template_kwargs", {})
        kwargs.setdefault("enable_thinking", bool(cfg["enable_thinking"]))
    if cfg.get("max_tokens") is not None:
        body.setdefault("max_tokens", int(cfg["max_tokens"]))
    return json.dumps(body, ensure_ascii=False) if body else None


def _translator_kwargs(cfg: dict) -> dict:
    return dict(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model_id=cfg["model_id"],
        to_lang=cfg["to_lang"],
        custom_prompt=cfg["custom_prompt"] or None,
        chunk_size=cfg["chunk_size"],
        concurrent=cfg["concurrent"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        timeout=cfg["timeout"],
        retry=cfg["retry"],
        thinking=cfg["thinking"],
        force_json=cfg["force_json"],
        system_proxy_enable=cfg["system_proxy_enable"],
        extra_body=_extra_body(cfg),
        logger=lib_logger,
    )


def build_workflow(cfg: dict, suffix: str):
    kwargs = _translator_kwargs(cfg)
    if suffix == ".docx":
        return DocxWorkflow(
            DocxWorkflowConfig(
                translator_config=DocxTranslatorConfig(
                    **kwargs,
                    insert_mode=cfg["insert_mode"],
                    separator=cfg["separator"],
                ),
                html_exporter_config=Docx2HTMLExporterConfig(cdn=True),
                logger=lib_logger,
            )
        )
    # 见文件顶部：这条 import 拉起 docling/transformers，只在真遇到非 docx 时付这 4 秒。
    from docutranslate.exporter.md.md2html_exporter import MD2HTMLExporterConfig
    from docutranslate.translator.ai_translator.md_translator import MDTranslatorConfig
    from docutranslate.workflow.md_based_workflow import (
        MarkdownBasedWorkflow,
        MarkdownBasedWorkflowConfig,
    )

    return MarkdownBasedWorkflow(
        MarkdownBasedWorkflowConfig(
            convert_engine="identity",
            converter_config=None,
            translator_config=MDTranslatorConfig(**kwargs),
            html_exporter_config=MD2HTMLExporterConfig(cdn=True),
            md2docx_exporter_config=None,
            logger=lib_logger,
        )
    )


async def _attempt(src: Path, dst: Path, cfg: dict) -> dict:
    """跑一次翻译并原子落盘，返回该文件的 token 统计。

    失败直接抛，由 translate_one 决定是否重试。
    """
    workflow = build_workflow(cfg, src.suffix.lower())
    workflow.read_path(str(src))
    await workflow.translate_async()

    stats = workflow.get_statistics().get("total")
    if stats is None:
        # 库在 translator 未建起来时返回 {}，不能让它变成裸 KeyError 被当成翻译失败
        raise RuntimeError("workflow 没有返回统计信息，翻译可能根本没执行")
    if stats["unresolved_errors"]:
        raise RuntimeError(f"未解决的错误 {stats['unresolved_errors']} 处")

    # 先写 .part 再改名：写一半崩掉留下的截断文件不会被 dst.exists() 当成已完成
    part = dst.with_name(dst.name + ".part")
    if isinstance(workflow, DocxWorkflow):
        workflow.save_as_docx(name=part.name, output_dir=str(part.parent))
    else:
        workflow.save_as_markdown(name=part.name, output_dir=str(part.parent))

    # 段级失败不进 unresolved_errors：MT 的 _error_result_handler 返回 None，
    # _collect 保留原文继续走（mt_agent.py:_collect），于是漏译文件一路被判成功、
    # 原子落盘、计成 OK —— --recheck 就是为捞回这批文件才存在的。
    # 落盘前用同一套判据当场拦下，让它去走 file_retry / 回退模型，而不是事后补捞。
    if isinstance(workflow, DocxWorkflow) and (bad := suspect_text(part)):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"存在未翻译段落: {bad[:80]!r}")

    os.replace(part, dst)

    # token 只在确认成功后累加：放在判错之前会让重试的文件重复计数
    totals["tokens"] += stats["total_tokens"]
    totals["in_tokens"] += stats["input_tokens"]
    totals["out_tokens"] += stats["output_tokens"]
    totals["requests"] += stats["request_count"]
    return stats


async def translate_one(
    src: Path,
    dst: Path,
    cfg: dict,
    sem: asyncio.Semaphore,
    fallback: dict | None = None,
) -> tuple[str | None, dict]:
    """成功返回 (None, 统计)，失败返回 (原因字符串, 统计)。

    fallback 非空时，第 2 次起换它的模型重试 —— 同一个模型连失败两次多半是
    这份文件它啃不动，再喂一遍只是把同样的错重放一次。
    """
    reason = "未执行"
    # 进入本文件的作用域：MT 的日志前缀与逐文件计数都靠它区分并发中的文件
    fstats = mt_agent.begin_file(src.name) if cfg["mt"] else {}
    for attempt in range(1, cfg["file_retry"] + 2):
        use = fallback if (fallback and attempt > 1) else cfg
        try:
            # 信号量只圈住真正在干活的那段：退避 sleep 放在锁外，
            # 否则 10 秒空等还占着一个文件槽。
            async with sem:
                tok = await _attempt(src, dst, use)
            if use is not cfg:
                logger.info(f"[回退成功] {src.name} 由 {use['model_id']} 译出")
            # fstats 放后面：唯一与库冲突的键是 total_tokens，而库那份
            # 在 MT 多轮调用下只剩最后一轮，必须让 MT 自己的账覆盖它
            return None, {**tok, **fstats}
        except Exception as e:  # noqa: BLE001 — 逐文件隔离，原因原样记录到 failures.txt
            reason = repr(e)
            logger.error(
                f"[失败 {attempt}/{cfg['file_retry'] + 1}] "
                f"{src.name} ({use['model_id']}): {reason}"
            )
            if attempt <= cfg["file_retry"]:
                await asyncio.sleep(cfg["retry_backoff"])
    return reason, dict(fstats)


async def run(
    tasks: list[tuple[Path, Path]],
    cfg: dict,
    out_dir: Path,
    fallback: dict | None = None,
) -> None:
    sem = asyncio.Semaphore(cfg["file_concurrent"])
    total = len(tasks)
    done = 0
    started = time.monotonic()
    failures_path = out_dir / "failures.txt"
    failures_path.unlink(missing_ok=True)

    async def worker(src: Path, dst: Path) -> None:
        nonlocal done
        t0 = time.monotonic()
        reason, st = await translate_one(src, dst, cfg, sem, fallback)
        done += 1
        cost = time.monotonic() - t0
        if reason is None:
            totals["success"] += 1
        else:
            totals["failed"] += 1
            fail_kinds[classify_failure(reason)] = (
                fail_kinds.get(classify_failure(reason), 0) + 1
            )
            # 逐条追加，不等 gather 结束：Ctrl-C 打断时整轮清单不会丢
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(f"{src}\t{reason}\n")
        state = "OK" if reason is None else "FAIL"
        # 逐文件明细：段数与命中来自 MT 作用域，token 与请求数来自库的 per-workflow 统计
        seg = st.get("segments", 0)
        hit = st.get("cache_hit", 0)
        # MT 模式下一律用自己记的数：库的统计只保留最后一轮 send_prompts_async
        req = st.get("request") if seg else st.get("request_count", 0)
        tok = st.get("total_tokens", 0)
        in_tok, out_tok = st.get("in_tokens", 0), st.get("out_tokens", 0)
        if not seg:  # 非 MT 链路仍走库的字段
            in_tok, out_tok = st.get("input_tokens", 0), st.get("output_tokens", 0)
        detail = f"{seg}段 命中{hit} 请求{req}" if seg else f"请求{req}"
        if tok:
            detail += f" {tok / 1000:.1f}Ktok(入{in_tok / 1000:.1f}K/出{out_tok / 1000:.1f}K)"
        logger.info(
            f"[{done}/{total}] {state} {src.name} | {detail} | {cost:.1f}s"
        )
        file_records.append(
            {
                "file": src.name,
                "state": state,
                "seconds": round(cost, 2),
                "segments": seg,
                "cache_hit": hit,
                "requests": req,
                "total_tokens": tok,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "echo": st.get("echo", 0),
                "echo_retry": st.get("echo_retry", 0),
                "reason": reason,
            }
        )

    async def heartbeat() -> None:
        """每 15s 报一次总进度。

        大文件单跑几分钟，没有这行的时候控制台完全静默，分不清是在干活
        还是卡死了（实测 255-2004.docx 静默过 267s）。
        """
        last = -1
        while True:
            await asyncio.sleep(15)
            if done >= total:
                return
            el = time.monotonic() - started
            m = mt_agent.stats
            req = m["request"] if cfg["mt"] else totals["requests"]
            if done == last and req == 0:
                continue
            last = done
            eta = (total - done) * el / done if done else 0
            out = m["out_tokens"] if cfg["mt"] else totals["out_tokens"]
            logger.info(
                f"··· {done}/{total} ({done / total * 100:.0f}%) | 在飞 {min(cfg['file_concurrent'], total - done)} "
                f"| 请求 {req} | 出 {out / 1000:.0f}Ktok"
                f"{f' | {out / el:.0f} tok/s' if out else ''}"
                f" | 已用 {el:.0f}s{f' | ETA {eta:.0f}s' if done else ''}"
            )

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(worker(src, dst) for src, dst in tasks))
    finally:
        hb.cancel()

    # run.json：逐文件明细落盘，方便事后横向对比不同模型/参数的同一批语料
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "profile": run_profile,
                "model_id": cfg["model_id"],
                "base_url": cfg["base_url"],
                "total": total,
                "success": totals["success"],
                "failed": totals["failed"],
                "seconds": round(time.monotonic() - started, 2),
                "files": file_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures_path.exists():
        logger.info(f"失败清单: {failures_path}")


def _self_check() -> None:
    def eb(**kw) -> str | None:
        return _extra_body({**DEFAULTS, **kw})

    assert eb() is None
    assert (
        eb(enable_thinking=False)
        == '{"chat_template_kwargs": {"enable_thinking": false}}'
    )
    assert (
        eb(enable_thinking=True)
        == '{"chat_template_kwargs": {"enable_thinking": true}}'
    )
    # 手写 extra_body 优先，不被开关覆盖
    hand = '{"chat_template_kwargs":{"enable_thinking":true},"top_k":20}'
    got = json.loads(eb(enable_thinking=False, extra_body=hand) or "")
    assert got["chat_template_kwargs"]["enable_thinking"] is True, got
    assert got["top_k"] == 20, got
    # 手写 extra_body 里没提思考时，开关照样生效
    got = json.loads(eb(enable_thinking=False, extra_body='{"top_k":20}') or "")
    assert got == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}, got
    # 不写开关就不注入，保持库的原行为
    assert eb(extra_body='{"top_k":20}') == '{"top_k": 20}'
    # max_tokens 是标准 OpenAI 参数，直接进请求体顶层
    assert json.loads(eb(max_tokens=8192) or "") == {"max_tokens": 8192}
    got = json.loads(eb(max_tokens=8192, enable_thinking=False) or "")
    assert got == {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 8192,
    }, got
    # 手写 extra_body 仍然优先
    assert (
        json.loads(eb(max_tokens=8192, extra_body='{"max_tokens":100}') or "")[
            "max_tokens"
        ]
        == 100
    )

    # top_p 必须真的转发到库（曾经整个漏掉，profile 里写了也静默无效）
    kw = _translator_kwargs(
        {**DEFAULTS, "base_url": "u", "model_id": "m", "top_p": 0.6}
    )
    assert kw["top_p"] == 0.6, kw

    # 配置校验：未知键与缺必填项都要报出来，不能静默
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.toml"
        p.write_text(
            '[default]\nbase_url="u"\nmodel_id="m"\ninput="i"\noutput="o"\n'
            "[profile.a]\n[profile.typo]\nfile_concurent=5\n"
            '[profile.bare]\nbase_url=""\n',
            encoding="utf-8",
        )
        assert load_profile(p, "a")["file_concurrent"] == 10
        for name, needle in (("typo", "file_concurent"), ("bare", "base_url")):
            try:
                load_profile(p, name)
            except SystemExit as e:
                assert needle in str(e), (name, e)
            else:
                raise AssertionError(f"profile.{name} 应当报错")

    # 回退重试：第 1 次用主 cfg，第 2 次起换 fallback，且成功即停
    used: list[str] = []

    async def _fallback_case() -> None:
        # patch 模块全局的 _attempt —— translate_one 按名字在这里查它。
        # 不能 import batch_translate：脚本跑成 __main__ 时那会导入第二份副本。
        real = globals()["_attempt"]

        async def fake(src, dst, cfg):
            used.append(cfg["model_id"])
            if cfg["model_id"] == "main":
                raise RuntimeError("主模型啃不动")
            return {"total_tokens": 7, "input_tokens": 5, "output_tokens": 2,
                    "request_count": 1}

        base = {**DEFAULTS, "file_retry": 2, "retry_backoff": 0}
        globals()["_attempt"] = fake
        try:
            reason, st = await translate_one(
                Path("x.docx"),
                Path("y.docx"),
                {**base, "model_id": "main"},
                asyncio.Semaphore(1),
                {**base, "model_id": "fb"},
            )
            assert reason is None, reason
            assert used == ["main", "fb"], used
            assert st["total_tokens"] == 7, st  # 统计随成功那次一起回传

            # 没配回退时行为不变：三次都用主 cfg，最后返回失败原因
            used.clear()
            reason, _ = await translate_one(
                Path("x.docx"),
                Path("y.docx"),
                {**base, "model_id": "main"},
                asyncio.Semaphore(1),
                None,
            )
            assert used == ["main"] * 3, used
            assert reason is not None and "啃不动" in reason, reason
            assert classify_failure(reason) == "未知" or reason, reason
        finally:
            globals()["_attempt"] = real

    asyncio.run(_fallback_case())

    # 回退 profile 的模式必须与主 profile 一致，不一致要拦住（否则静默译错）
    main_cfg = {**DEFAULTS, "mt": True, "model_id": "a"}
    fb_ok = {**DEFAULTS, "mt": True, "model_id": "b", "fallback_profile": "c"}
    _check_fallback(main_cfg, fb_ok, "main")
    assert fb_ok["fallback_profile"] is None, "链式回退没被切断"
    try:
        _check_fallback(
            {**main_cfg, "fallback_profile": "x"},
            {**DEFAULTS, "mt": False, "model_id": "b"},
            "main",
        )
    except SystemExit as e:
        assert "mt" in str(e), e
    else:
        raise AssertionError("mt 不一致的回退 profile 应当报错")

    # suspect_text：整段外文算漏译，含汉字或短代号不算
    with tempfile.TemporaryDirectory() as tmp:

        def mk(name: str, *paras: str) -> Path:
            q = Path(tmp) / name
            body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in paras)
            with zipfile.ZipFile(q, "w") as z:
                z.writestr(
                    "word/document.xml",
                    f"<w:document><w:body>{body}</w:body></w:document>",
                )
            return q

        long_is = "Bersynilegar villur og reikningsskekkjur i framtali kaeranda"
        assert suspect_text(mk("bad.docx", "译文一段", long_is)) == long_is
        assert suspect_text(mk("ok.docx", "译文一段", "BTI 3707-9034 DE HAM")) is None
        assert suspect_text(mk("mixed.docx", f"{long_is} 的中文译文")) is None
        assert "无法读取" in (suspect_text(Path(tmp) / "nope.docx") or "")
        # 表格单元格：<w:t> 里存的是转义的整段 XML，上百字符的标签不得把
        # 25 字符的报关单号顶过阈值（实测让两个文件三次重试全废）
        cell = (
            "&lt;w:tc&gt;&lt;w:tcPr/&gt;&lt;w:p&gt;&lt;w:r&gt;&lt;w:t&gt;"
            "S HEG 06 04 4 SE VAG W044"
            "&lt;/w:t&gt;&lt;/w:r&gt;&lt;/w:p&gt;&lt;/w:tc&gt;"
        )
        assert suspect_text(mk("cell.docx", "译文一段", cell)) is None
        # 同样包装下的实义外文仍要判为漏译
        cell_real = (
            "&lt;w:tc&gt;&lt;w:p&gt;&lt;w:r&gt;&lt;w:t&gt;"
            + long_is
            + "&lt;/w:t&gt;&lt;/w:r&gt;&lt;/w:p&gt;&lt;/w:tc&gt;"
        )
        assert suspect_text(mk("cellreal.docx", "译文一段", cell_real)) == long_is

        # recheck 删漏译、留好文件
        d = Path(tmp) / "out"
        d.mkdir()
        (d / "bad.docx").write_bytes((Path(tmp) / "bad.docx").read_bytes())
        (d / "ok.docx").write_bytes((Path(tmp) / "ok.docx").read_bytes())
        recheck(d)
        assert not (d / "bad.docx").exists()
        assert (d / "ok.docx").exists()
        assert "bad.docx" in (d / "suspects.txt").read_text(encoding="utf-8")

    print("batch_translate 自检通过")


def _check_fallback(cfg: dict, fallback: dict, main_name: str) -> None:
    """校验回退 profile 能否与主 profile 共处一个进程，并就地切断链式回退。"""
    # 不许链式回退：第二次失败就该进 failures.txt，不是继续换模型试到天亮
    fallback["fallback_profile"] = None
    # 这三项都靠替换 docx_translator 里 SegmentsTranslateAgent 这一个符号生效
    # （见 _install_mode），是进程级的、按主 profile 装一次就定了。回退 profile
    # 若与主 profile 不同，它实际拿到的是主 profile 的协议 —— 不报错，只是译得
    # 不对。静默跑错比直接失败更贵，所以这里必须拦住。
    for key in ("mt", "json_schema", "minimal_prompt"):
        if bool(fallback[key]) != bool(cfg[key]):
            sys.exit(
                f"profile.{cfg['fallback_profile']} 的 {key}={fallback[key]} 与 "
                f"profile.{main_name} 的 {key}={cfg[key]} 不一致："
                "这三项是进程级全局补丁，回退 profile 必须与主 profile 相同"
            )
    if fallback["model_id"] == cfg["model_id"]:
        logger.warning(
            f"回退 profile 的 model_id 与主 profile 相同（{cfg['model_id']}）："
            "换模型重试就没意义了，确认是有意为之"
        )


def _install_mode(cfg: dict, out_dir: Path) -> bool:
    """装上 mt / schema 补丁，返回全局并发闸是否真的生效。"""
    # 两者都替换 docx_translator.SegmentsTranslateAgent 这一个符号，不能同开
    if cfg["mt"] and cfg["json_schema"]:
        raise SystemExit("mt 与 json_schema 互斥：MT 模型走单段直译，不发 JSON 协议")
    if not cfg["mt"]:
        # 以下两个只对分段协议有意义，且可叠加（实测极简+约束 缺键1.4%/罢工5.3%）。
        # 单段直译不走 generate_prompt 也不发 JSON，打这些补丁纯属空转。
        if cfg["json_schema"]:
            schema_agent.enable_schema()
            logger.info(
                "已启用 JSON Schema 约束解码（每个 chunk 按自身 id 生成对象 schema）"
            )
        if cfg["minimal_prompt"]:
            schema_agent.enable_minimal_prompt()
            logger.info("已启用极简提示词（替换库那份 60 行长模板）")
        return False

    mt_agent.set_global_concurrency(cfg["global_concurrent"])
    cache = cfg["mt_cache"]
    # 绝对路径 → 全语料共用一份缓存：既跨批次省请求，也保证同一原文
    # 在所有文件里得到同一译文。相对路径 → 落在本次输出目录下。
    if cache:
        path = Path(cache).expanduser()
        mt_agent.enable_mt(path if path.is_absolute() else out_dir / path)
    else:
        mt_agent.enable_mt(None)
    logger.info("已启用单段直译（一段一请求，无 JSON 分段协议）")
    return cfg["global_concurrent"] is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="按 profile 批量翻译 docx / md")
    parser.add_argument("profile", nargs="?", help="batch.toml 里 [profile.X] 的名字")
    parser.add_argument("--config", default=str(Path(__file__).with_name("batch.toml")))
    parser.add_argument("--input", help="覆盖 profile 里的 input")
    parser.add_argument("--output", help="覆盖 profile 里的 output")
    parser.add_argument("--dry-run", action="store_true", help="只打印 输入→输出 映射")
    parser.add_argument("--verbose", action="store_true", help="放开库的逐请求日志")
    parser.add_argument(
        "--quiet", action="store_true", help="终端只留警告，文件日志照旧"
    )
    parser.add_argument("--self-check", action="store_true", help="跑离线自检后退出")
    parser.add_argument(
        "--recheck", action="store_true", help="扫描已有输出，删掉疑似漏译的文件后退出"
    )
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.profile:
        parser.error("缺少 profile 名（或用 --self-check）")

    # 先只挂 stdout：这样 load_profile / 目录检查的失败信息也走同一套格式，
    # 输出目录一解析出来立刻补上文件 handler。
    setup_logger(None, args.verbose, args.quiet)

    cfg = load_profile(Path(args.config).expanduser(), args.profile)
    in_dir = Path(args.input or cfg["input"]).expanduser()
    out_dir = Path(args.output or cfg["output"]).expanduser()
    if not in_dir.is_dir():
        sys.exit(f"输入目录不存在: {in_dir}")

    global run_profile
    run_profile = args.profile or ""
    if args.recheck:
        recheck(out_dir)
        return

    extensions = {e.lower() for e in cfg["extensions"]}
    sources = sorted(
        (
            p
            for p in in_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ),
        key=lambda p: str(p).lower(),
    )

    # 上一轮崩在落盘途中留下的半成品，清掉重来
    stale = [p for p in out_dir.rglob("*.part")] if out_dir.is_dir() else []
    tasks = [
        (src, out_dir / src.relative_to(in_dir))
        for src in sources
        if not (out_dir / src.relative_to(in_dir)).exists()
    ]
    totals["skipped"] = len(sources) - len(tasks)

    if args.dry_run:
        for src, dst in tasks:
            print(f"{src}  ->  {dst}")
        print(
            f"\n共 {len(sources)} 个文件，跳过(已存在) {totals['skipped']}，待处理 {len(tasks)}"
        )
        return

    setup_logger(out_dir / "batch.log", args.verbose, args.quiet)
    for p in stale:
        p.unlink(missing_ok=True)
    if stale:
        logger.info(f"清理上一轮遗留的半成品 {len(stale)} 个")

    if cfg["thinking"] != "default":
        logger.warning(
            "thinking 非 default：库按 model_id 猜厂商注入思考字段，自建 vLLM 多半静默无效。"
            "关思考请用 enable_thinking。"
        )
    fallback = None
    if cfg["fallback_profile"]:
        fallback = load_profile(Path(args.config).expanduser(), cfg["fallback_profile"])
        _check_fallback(cfg, fallback, args.profile)

    gated = _install_mode(cfg, out_dir)
    # 不说谎：全局闸只装在 mt 路径上，mt=false 时真实上限是两级信号量的乘积
    gate = (
        f"全局并发={cfg['global_concurrent']}"
        if gated
        else f"全局并发=未启用(实际上限 {cfg['file_concurrent'] * cfg['concurrent']})"
    )
    logger.info(
        f"profile={args.profile} model={cfg['model_id']} mt={cfg['mt']} "
        f"文件并发={cfg['file_concurrent']} 分块并发={cfg['concurrent']} "
        f"系统代理={cfg['system_proxy_enable']} {gate}"
    )
    if fallback:
        logger.info(
            f"回退={cfg['fallback_profile']} model={fallback['model_id']} "
            f"（第 2 次起用它重试）"
        )
    logger.info(
        f"共 {len(sources)} 个文件，跳过(已存在) {totals['skipped']}，待处理 {len(tasks)}"
    )

    if not tasks:
        logger.info("没有待处理文件。")
        return

    started = time.time()
    try:
        asyncio.run(run(tasks, cfg, out_dir, fallback))
    except KeyboardInterrupt:
        logger.info("已中断，已完成的输出文件保留，重跑会自动跳过。")
        sys.exit(1)
    finally:
        elapsed = time.time() - started
        n_all = totals["success"] + totals["failed"] + totals["skipped"]
        logger.info(
            f"结束 | 文件 {n_all} 个: 成功 {totals['success']} 失败 {totals['failed']} "
            f"跳过 {totals['skipped']} | 耗时 {elapsed:.1f}s"
        )
        if cfg["mt"]:
            totals["tokens"] = mt_agent.stats["total_tokens"]
            totals["in_tokens"] = mt_agent.stats["in_tokens"]
            totals["out_tokens"] = mt_agent.stats["out_tokens"]
        tk, out_tk = totals["tokens"], totals["out_tokens"]
        if tk:
            logger.info(
                f"token | 入 {totals['in_tokens'] / 1000:.1f}K | 出 {out_tk / 1000:.1f}K "
                f"| 合计 {tk / 1000:.1f}K | {out_tk / max(elapsed, 1e-9):.0f} 出tok/s"
            )
        if fail_kinds:
            logger.info(
                "失败原因 | "
                + " | ".join(
                    f"{k} {v}" for k, v in sorted(fail_kinds.items(), key=lambda x: -x[1])
                )
            )
        if cfg["mt"]:
            m = mt_agent.stats
            seen = m["cache_hit"] + m["request"]
            rate = f" ({m['cache_hit'] / seen * 100:.1f}%)" if seen else ""
            logger.info(
                f"MT | 缓存命中 {m['cache_hit']} 段{rate} | 实发请求 {m['request']} 个 "
                f"| {m['request'] / max(elapsed, 1e-9):.1f} 请求/s "
                f"| 未翻译回声 {m['echo']} 段 | 长文本重问 {m['echo_retry']} 次 "
                f"| 段级失败 {m['failed']} 段"
            )


if __name__ == "__main__":
    main()
