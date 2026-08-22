#!/usr/bin/env python3
"""按 profile 批量翻译目录下的 docx / md。

    python examples/batch_translate.py hunyuan
    python examples/batch_translate.py qwen --input A --output B
    python examples/batch_translate.py hunyuan --dry-run

已存在的输出文件直接跳过，失败文件记入 <输出目录>/failures.txt，重跑时自动重试。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import tomllib
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mt_agent  # noqa: E402
import schema_agent  # noqa: E402
from docutranslate.exporter.docx.docx2html_exporter import Docx2HTMLExporterConfig  # noqa: E402
from docutranslate.exporter.md.md2html_exporter import MD2HTMLExporterConfig  # noqa: E402
from docutranslate.translator.ai_translator.docx_translator import DocxTranslatorConfig  # noqa: E402
from docutranslate.translator.ai_translator.md_translator import MDTranslatorConfig  # noqa: E402
from docutranslate.workflow.docx_workflow import DocxWorkflow, DocxWorkflowConfig  # noqa: E402
from docutranslate.workflow.md_based_workflow import (  # noqa: E402
    MarkdownBasedWorkflow,
    MarkdownBasedWorkflowConfig,
)

logger = logging.getLogger("batch")
totals = {"success": 0, "skipped": 0, "failed": 0, "tokens": 0}


def load_profile(config_path: Path, name: str) -> dict:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    profiles = raw.get("profile", {})
    if name not in profiles:
        available = ", ".join(profiles) or "(无)"
        sys.exit(f"{config_path} 里没有 profile.{name}，可用: {available}")
    return {**raw.get("default", {}), **profiles[name]}


def setup_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


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
    if "enable_thinking" in cfg:
        kwargs = body.setdefault("chat_template_kwargs", {})
        kwargs.setdefault("enable_thinking", bool(cfg["enable_thinking"]))
    if "max_tokens" in cfg:
        body.setdefault("max_tokens", int(cfg["max_tokens"]))
    return json.dumps(body, ensure_ascii=False) if body else None


def _translator_kwargs(cfg: dict) -> dict:
    return dict(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model_id=cfg["model_id"],
        to_lang=cfg["to_lang"],
        custom_prompt=cfg.get("custom_prompt") or None,
        chunk_size=cfg["chunk_size"],
        concurrent=cfg["concurrent"],
        temperature=cfg["temperature"],
        timeout=cfg["timeout"],
        retry=cfg["retry"],
        thinking=cfg.get("thinking", "default"),
        force_json=cfg.get("force_json", False),
        system_proxy_enable=cfg.get("system_proxy_enable", False),
        extra_body=_extra_body(cfg),
        logger=logger,
    )


def build_workflow(cfg: dict, suffix: str):
    kwargs = _translator_kwargs(cfg)
    if suffix == ".docx":
        return DocxWorkflow(
            DocxWorkflowConfig(
                translator_config=DocxTranslatorConfig(
                    **kwargs,
                    insert_mode=cfg.get("insert_mode", "replace"),
                    separator=cfg.get("separator", "\n"),
                ),
                html_exporter_config=Docx2HTMLExporterConfig(cdn=True),
                logger=logger,
            )
        )
    return MarkdownBasedWorkflow(
        MarkdownBasedWorkflowConfig(
            convert_engine="identity",
            converter_config=None,
            translator_config=MDTranslatorConfig(**kwargs),
            html_exporter_config=MD2HTMLExporterConfig(cdn=True),
            md2docx_exporter_config=None,
            logger=logger,
        )
    )


async def translate_one(
    src: Path, dst: Path, cfg: dict, sem: asyncio.Semaphore
) -> str | None:
    """成功返回 None，失败返回原因字符串。"""
    suffix = src.suffix.lower()
    backoff = cfg.get("retry_backoff", 10)
    reason = "未执行"
    async with sem:
        for attempt in range(1, cfg["file_retry"] + 2):
            try:
                workflow = build_workflow(cfg, suffix)
                workflow.read_path(str(src))
                await workflow.translate_async()

                stats = workflow.get_statistics()["total"]
                totals["tokens"] += stats["total_tokens"]
                if stats["unresolved_errors"]:
                    raise RuntimeError(f"未解决的错误 {stats['unresolved_errors']} 处")

                if isinstance(workflow, DocxWorkflow):
                    workflow.save_as_docx(name=dst.name, output_dir=str(dst.parent))
                else:
                    workflow.save_as_markdown(name=dst.name, output_dir=str(dst.parent))
                return None
            except Exception as e:  # noqa: BLE001 — 逐文件隔离，原因原样记录到 failures.txt
                reason = repr(e)
                logger.error(
                    f"[失败 {attempt}/{cfg['file_retry'] + 1}] {src.name}: {reason}"
                )
                if attempt <= cfg["file_retry"]:
                    await asyncio.sleep(backoff)
    return reason


async def run(tasks: list[tuple[Path, Path]], cfg: dict, out_dir: Path) -> None:
    sem = asyncio.Semaphore(cfg["file_concurrent"])
    total = len(tasks)
    done = 0
    failures: list[str] = []

    async def worker(src: Path, dst: Path) -> None:
        nonlocal done
        reason = await translate_one(src, dst, cfg, sem)
        done += 1
        if reason is None:
            totals["success"] += 1
        else:
            totals["failed"] += 1
            failures.append(f"{src}\t{reason}")
        state = "OK" if reason is None else "FAIL"
        logger.info(f"[{done}/{total}] {state} {src.name}")

    await asyncio.gather(*(worker(src, dst) for src, dst in tasks))

    failures_path = out_dir / "failures.txt"
    if failures:
        failures_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        logger.info(f"失败清单: {failures_path}")
    elif failures_path.exists():
        failures_path.unlink()


def _self_check() -> None:
    assert _extra_body({}) is None
    assert (
        _extra_body({"enable_thinking": False})
        == '{"chat_template_kwargs": {"enable_thinking": false}}'
    )
    assert (
        _extra_body({"enable_thinking": True})
        == '{"chat_template_kwargs": {"enable_thinking": true}}'
    )
    # 手写 extra_body 优先，不被开关覆盖
    hand = '{"chat_template_kwargs":{"enable_thinking":true},"top_k":20}'
    got = json.loads(_extra_body({"enable_thinking": False, "extra_body": hand}) or "")
    assert got["chat_template_kwargs"]["enable_thinking"] is True, got
    assert got["top_k"] == 20, got
    # 手写 extra_body 里没提思考时，开关照样生效
    got = json.loads(
        _extra_body({"enable_thinking": False, "extra_body": '{"top_k":20}'}) or ""
    )
    assert got == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}, got
    # 不写开关就不注入，保持库的原行为
    assert _extra_body({"extra_body": '{"top_k":20}'}) == '{"top_k": 20}'
    # max_tokens 是标准 OpenAI 参数，直接进请求体顶层
    assert json.loads(_extra_body({"max_tokens": 8192}) or "") == {"max_tokens": 8192}
    got = json.loads(_extra_body({"max_tokens": 8192, "enable_thinking": False}) or "")
    assert got == {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 8192,
    }, got
    # 手写 extra_body 仍然优先
    assert (
        json.loads(
            _extra_body({"max_tokens": 8192, "extra_body": '{"max_tokens":100}'}) or ""
        )["max_tokens"]
        == 100
    )
    print("batch_translate 自检通过")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 profile 批量翻译 docx / md")
    parser.add_argument("profile", help="batch.toml 里 [profile.X] 的名字")
    parser.add_argument("--config", default=str(Path(__file__).with_name("batch.toml")))
    parser.add_argument("--input", help="覆盖 profile 里的 input")
    parser.add_argument("--output", help="覆盖 profile 里的 output")
    parser.add_argument("--dry-run", action="store_true", help="只打印 输入→输出 映射")
    parser.add_argument("--self-check", action="store_true", help="跑离线自检后退出")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return

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

    tasks: list[tuple[Path, Path]] = []
    for src in sources:
        dst = out_dir / src.relative_to(in_dir)
        if dst.exists():
            totals["skipped"] += 1
        else:
            tasks.append((src, dst))

    if args.dry_run:
        for src, dst in tasks:
            print(f"{src}  ->  {dst}")
        print(
            f"\n共 {len(sources)} 个文件，跳过(已存在) {totals['skipped']}，待处理 {len(tasks)}"
        )
        return

    setup_logger(out_dir / "batch.log")
    logger.info(
        f"profile={args.profile} model={cfg['model_id']} mt={bool(cfg.get('mt'))} "
        f"文件并发={cfg['file_concurrent']} 分块并发={cfg['concurrent']} "
        f"系统代理={bool(cfg.get('system_proxy_enable', False))} "
        f"全局并发={cfg.get('global_concurrent')}"
    )
    if cfg.get("thinking", "default") != "default":
        logger.warning(
            "thinking 非 default：库按 model_id 猜厂商注入思考字段，自建 vLLM 多半静默无效。"
            "关思考请用 enable_thinking。"
        )
    logger.info(
        f"共 {len(sources)} 个文件，跳过(已存在) {totals['skipped']}，待处理 {len(tasks)}"
    )

    # 两者都替换 docx_translator.SegmentsTranslateAgent 这一个符号，不能同开
    if cfg.get("mt") and cfg.get("json_schema"):
        raise SystemExit("mt 与 json_schema 互斥：MT 模型走单段直译，不发 JSON 协议")
    if cfg.get("mt"):
        mt_agent.set_global_concurrency(cfg.get("global_concurrent"))
        cache = cfg.get("mt_cache")
        # 绝对路径 → 全语料共用一份缓存：既跨批次省请求，也保证同一原文
        # 在所有文件里得到同一译文。相对路径 → 落在本次输出目录下。
        if cache:
            path = Path(cache).expanduser()
            mt_agent.enable_mt(path if path.is_absolute() else out_dir / path)
        else:
            mt_agent.enable_mt(None)
        logger.info("已启用单段直译（一段一请求，无 JSON 分段协议）")
    else:
        # 以下两个只对分段协议有意义，且可叠加（实测极简+约束 缺键1.4%/罢工5.3%）。
        # 单段直译不走 generate_prompt 也不发 JSON，打这些补丁纯属空转。
        if cfg.get("json_schema"):
            schema_agent.enable_schema()
            logger.info(
                "已启用 JSON Schema 约束解码（每个 chunk 按自身 id 生成对象 schema）"
            )
        if cfg.get("minimal_prompt"):
            schema_agent.enable_minimal_prompt()
            logger.info("已启用极简提示词（替换库那份 60 行长模板）")

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
        if cfg.get("mt"):
            m = mt_agent.stats
            logger.info(
                f"MT | 缓存命中 {m['cache_hit']} 段 | 实发请求 {m['request']} 个 "
                f"| 未翻译回声 {m['echo']} 段 | 段级失败 {m['failed']} 段"
            )


if __name__ == "__main__":
    main()
