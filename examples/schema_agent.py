"""用 JSON Schema 约束解码消灭 docx 分段协议的缺键。

分段协议要求模型一边翻译一边维护 id 账本，9B 级模型扛不住：实测缺键率
47.5%（不约束）/ 2.1%（response_format=json_object）。json_object 只保证
语法合法，键齐不齐靠运气；把 required 写死全部原文 id 的对象 schema 则是
解码器层面的硬保证——vLLM 的约束解码不可能吐出缺 required 键的对象。

库源码零改动：唯一耦合点是 enable_schema() 里替换的那一个符号。
需要后端支持 response_format={"type":"json_schema"}（vLLM ≥0.6 / xgrammar）。
"""

from __future__ import annotations

import json

from docutranslate.agents.agent import PartialAgentResultError
from docutranslate.agents.segments_agent import (
    SegmentsTranslateAgent,
    get_original_segments,
)

# 协议原文让模型输出数组 [{"id":..,"t":..}]，schema 约束的是对象；
# 指令与约束冲突时解码器赢，但把指令也改掉能少绕一圈。
_OBJECT_HINT = (
    "\nOutput a JSON object mapping every input ID to its translation, "
    'e.g. {"3":"译文3","4":"译文4"}. Every input ID must appear as a key.\n'
)


# 冰岛语→中文实测 1675 组配对：长度比中位 0.37，p99 1.00，最大 1.28。
# 2 倍加 64 比实测最坏情况仍宽 56%，同时把复读循环掐死在语法层。
_LEN_FACTOR = 2
_LEN_SLACK = 64


def cap_of(source: str) -> int:
    return len(source) * _LEN_FACTOR + _LEN_SLACK


def schema_for(chunk: dict[str, str]) -> dict:
    """每个值单独限长。

    required 堵的是缺键，maxLength 堵的是复读：schema 只约束 JSON 骨架，
    字符串值内部无约束，模型在值里进复读循环时语法始终合法，能一路生成到
    上下文上限（实测撞出过单次 330302 字符）。截断后库会走"继续获取"，
    而续写请求带不了对象 schema，缺键保证就在那里破掉。
    """
    return {
        "type": "object",
        "properties": {
            k: {"type": "string", "maxLength": cap_of(v)} for k, v in chunk.items()
        },
        "required": list(chunk),
        "additionalProperties": False,
    }


def chunk_of(prompt: str) -> dict[str, str] | None:
    """从 prompt 的 <input> 块里取出本次 chunk 的原文（id → 原文）。

    续写请求（截断后 continue）的 prompt 里没有 <input> 块，返回 None 表示
    这一发不加 schema —— 续写只补数组尾巴，套对象 schema 反而会打架。
    """
    try:
        raw = get_original_segments(prompt)
    except ValueError:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    return {str(k): str(v) for k, v in parsed.items()}


class SchemaSegmentsAgent(SegmentsTranslateAgent):
    """每个 chunk 带一份按自己 id 生成的对象 schema。"""

    def _repair_values(self, chunk: dict, origin_prompt: str, logger) -> None:
        """两种 schema 特有的坏值退回原文，各出一条 warning。

        空串：schema 逼模型给每个 key 填值，没得翻的段（"II." 这类）会被填成
        空串，写进 docx 就是把原文抹掉。

        撞上限：值长度正好等于 maxLength 说明是被语法掐断的，几乎只可能是
        复读循环（实测正常译文最长才 1.28 倍原文，上限给的是 3 倍）。掐断的
        半截译文同样不能进文档。

        都不静默兜底 —— 日志里能数出来有多少段是这样。
        """
        try:
            original = json.loads(get_original_segments(origin_prompt))
        except (ValueError, json.JSONDecodeError):
            return
        if not isinstance(original, dict):
            return
        for key, value in list(chunk.items()):
            src = original.get(key)
            if src is None or not str(src).strip():
                continue
            text = str(value)
            if not text.strip():
                reason = "译文为空"
            elif len(text) >= cap_of(str(src)):
                reason = f"译文撞上限{cap_of(str(src))}字符，疑似复读被掐断"
            else:
                continue
            logger.warning(f"段 {key} {reason}，已退回原文: {str(src)[:40]!r}")
            chunk[key] = str(src)

    def _result_handler(self, result: str, origin_prompt: str, logger):
        try:
            out = super()._result_handler(result, origin_prompt, logger)
        except PartialAgentResultError as e:
            self._repair_values(e.partial_result, origin_prompt, logger)
            raise
        if isinstance(out, dict):
            self._repair_values(out, origin_prompt, logger)
        return out

    def _prepare_request_data(
        self,
        prompt: str,
        system_prompt: str,
        temperature=None,
        top_p=None,
        json_format=False,
    ):
        headers, data = super()._prepare_request_data(
            prompt, system_prompt, temperature, top_p, json_format
        )
        # ponytail: 每个 chunk 的 id 不同 → schema 不同 → xgrammar 的 grammar 缓存
        # 命不中，每发一次多 20-50ms 编译。实测每 chunk 12.2s→12.4s（1.6%），不值得动。
        # 真要压：把 id 在 chunk 内重编号成 0..N-1，schema 只随 N 变，缓存就能复用。
        chunk = chunk_of(prompt)
        if chunk is None:
            return headers, data
        # response_format 在父类里是最后写的，这里覆盖它即可（extra_body 已合并完）
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "segments",
                "schema": schema_for(chunk),
                "strict": True,
            },
        }
        for msg in data["messages"]:
            if msg["role"] == "user":
                msg["content"] += _OBJECT_HINT
        return headers, data


def enable_schema() -> None:
    """把 docx 链路的分段 Agent 换成带 schema 约束的实现。"""
    import docutranslate.translator.ai_translator.docx_translator as docx_translator

    docx_translator.SegmentsTranslateAgent = SchemaSegmentsAgent


def demo() -> None:
    from docutranslate.agents.segments_agent import generate_prompt

    chunk = {"3": "hello", "4": "world"}
    p = generate_prompt(json.dumps(chunk, ensure_ascii=False, indent=0), "简体中文")
    assert chunk_of(p) == chunk, chunk_of(p)

    s = schema_for({"3": "hello", "4": "x" * 100})
    assert s["required"] == ["3", "4"]
    assert s["additionalProperties"] is False
    # 每个值按自己原文长度限长，不是一刀切
    assert s["properties"]["3"]["maxLength"] == 5 * _LEN_FACTOR + _LEN_SLACK
    assert s["properties"]["4"]["maxLength"] == 100 * _LEN_FACTOR + _LEN_SLACK
    # 实测最长译文是原文的 1.28 倍，上限必须留足余量
    assert cap_of("x" * 100) > 100 * 1.28

    # 续写 prompt 没有 <input> 块 —— 必须优雅退回，不加 schema
    assert chunk_of("你之前的翻译输出被截断了。请继续。") is None
    # 空 chunk 不生成 schema
    assert chunk_of(generate_prompt("{}", "简体中文")) is None

    class _L:
        msgs: list = []

        def warning(self, m):
            _L.msgs.append(m)

    agent = SchemaSegmentsAgent.__new__(SchemaSegmentsAgent)

    # 空串必须退回原文，否则 docx 里的 "II." 会被抹掉
    p2 = generate_prompt(
        json.dumps({"1": "II.", "2": "hello"}, ensure_ascii=False), "简体中文"
    )
    got = {"1": "", "2": "你好"}
    agent._repair_values(got, p2, _L())
    assert got == {"1": "II.", "2": "你好"}, got
    assert len(_L.msgs) == 1 and "译文为空" in _L.msgs[0], _L.msgs

    # 撞上限 = 复读被语法掐断，同样退回原文
    _L.msgs.clear()
    src = "x" * 20
    p3 = generate_prompt(json.dumps({"1": src}, ensure_ascii=False), "简体中文")
    got = {"1": "复" * cap_of(src)}
    agent._repair_values(got, p3, _L())
    assert got == {"1": src}, got
    assert len(_L.msgs) == 1 and "复读" in _L.msgs[0], _L.msgs

    # 正常长度的译文不该被碰
    _L.msgs.clear()
    got = {"1": "正常译文"}
    agent._repair_values(got, p3, _L())
    assert got == {"1": "正常译文"} and not _L.msgs

    # 原文本身就是空的，不该报 warning
    _L.msgs.clear()
    p4 = generate_prompt(json.dumps({"1": ""}, ensure_ascii=False), "简体中文")
    got = {"1": ""}
    agent._repair_values(got, p4, _L())
    assert got == {"1": ""} and not _L.msgs

    print("schema_agent 自检通过")


if __name__ == "__main__":
    demo()
