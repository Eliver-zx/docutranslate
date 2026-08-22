#!/usr/bin/env python3
"""按 profile 批量翻译目录下的 docx / md。

    python examples/batch_translate.py hunyuan
    python examples/batch_translate.py qwen --input A --output B
    python examples/batch_translate.py hunyuan --dry-run

输出文件已存在就跳过。输出先写 <名字>.part、成功后原子改名，所以半成品不会被
当成已完成；启动时清理上一轮遗留的 .part。失败逐条追加到 <输出目录>/failures.txt
（纯报告：失败文件没有输出，重跑时靠"输出不存在"自然重试）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import tomllib
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

logger = logging.getLogger("batch")
# 库的日志走子 logger：它每个请求打一条"协程-已完成 N/M"，多文件并发下交织成噪音。
# 子 logger 默认压到 WARNING，--verbose 放开；handler 挂在父 logger 上，共用输出。
lib_logger = logging.getLogger("batch.lib")
totals = {"success": 0, "skipped": 0, "failed": 0, "tokens": 0}

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


async def _attempt(src: Path, dst: Path, cfg: dict) -> None:
    """跑一次翻译并原子落盘。失败直接抛，由 translate_one 决定是否重试。"""
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
    os.replace(part, dst)

    # token 只在确认成功后累加：放在判错之前会让重试的文件重复计数
    totals["tokens"] += stats["total_tokens"]


async def translate_one(
    src: Path, dst: Path, cfg: dict, sem: asyncio.Semaphore
) -> str | None:
    """成功返回 None，失败返回原因字符串。"""
    reason = "未执行"
    for attempt in range(1, cfg["file_retry"] + 2):
        try:
            # 信号量只圈住真正在干活的那段：退避 sleep 放在锁外，
            # 否则 10 秒空等还占着一个文件槽。
            async with sem:
                await _attempt(src, dst, cfg)
            return None
        except Exception as e:  # noqa: BLE001 — 逐文件隔离，原因原样记录到 failures.txt
            reason = repr(e)
            logger.error(
                f"[失败 {attempt}/{cfg['file_retry'] + 1}] {src.name}: {reason}"
            )
            if attempt <= cfg["file_retry"]:
                await asyncio.sleep(cfg["retry_backoff"])
    return reason


async def run(tasks: list[tuple[Path, Path]], cfg: dict, out_dir: Path) -> None:
    sem = asyncio.Semaphore(cfg["file_concurrent"])
    total = len(tasks)
    done = 0
    started = time.monotonic()
    failures_path = out_dir / "failures.txt"
    failures_path.unlink(missing_ok=True)

    async def worker(src: Path, dst: Path) -> None:
        nonlocal done
        t0 = time.monotonic()
        reason = await translate_one(src, dst, cfg, sem)
        done += 1
        if reason is None:
            totals["success"] += 1
        else:
            totals["failed"] += 1
            # 逐条追加，不等 gather 结束：Ctrl-C 打断时整轮清单不会丢
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(f"{src}\t{reason}\n")
        state = "OK" if reason is None else "FAIL"
        rate = done / max(time.monotonic() - started, 1e-9) * 60
        logger.info(
            f"[{done}/{total}] {state} {src.name} "
            f"({time.monotonic() - t0:.1f}s, {rate:.1f} 文件/分)"
        )

    await asyncio.gather(*(worker(src, dst) for src, dst in tasks))
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

    print("batch_translate 自检通过")


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
    logger.info(
        f"共 {len(sources)} 个文件，跳过(已存在) {totals['skipped']}，待处理 {len(tasks)}"
    )

    if not tasks:
        logger.info("没有待处理文件。")
        return

    started = time.time()
    try:
        asyncio.run(run(tasks, cfg, out_dir))
    except KeyboardInterrupt:
        logger.info("已中断，已完成的输出文件保留，重跑会自动跳过。")
        sys.exit(1)
    finally:
        elapsed = time.time() - started
        logger.info(
            f"结束 | 成功 {totals['success']} | 跳过 {totals['skipped']} | 失败 {totals['failed']} "
            f"| token {totals['tokens'] / 1000:.1f}K | 耗时 {elapsed:.1f}s"
        )
        if cfg["mt"]:
            m = mt_agent.stats
            logger.info(
                f"MT | 缓存命中 {m['cache_hit']} 段 | 实发请求 {m['request']} 个 "
                f"| 未翻译回声 {m['echo']} 段 | 段级失败 {m['failed']} 段"
            )


if __name__ == "__main__":
    main()
