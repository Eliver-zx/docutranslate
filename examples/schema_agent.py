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


def schema_for(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in ids},
        "required": ids,
        "additionalProperties": False,
    }


def ids_of(prompt: str) -> list[str] | None:
    """从 prompt 的 <input> 块里取出本次 chunk 的全部原文 id。

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
    return [str(k) for k in parsed]


class SchemaSegmentsAgent(SegmentsTranslateAgent):
    """每个 chunk 带一份按自己 id 生成的对象 schema。"""

    def _restore_blanks(self, chunk: dict, origin_prompt: str, logger) -> None:
        """schema 逼模型给每个 key 填值，没得翻的段（"II." 这类）会被填成空串。

        空串写进 docx 就是把原文抹掉。这里原样退回原文并出一条 warning ——
        不静默兜底，日志里能数出来有多少段是这样。
        """
        try:
            original = json.loads(get_original_segments(origin_prompt))
        except (ValueError, json.JSONDecodeError):
            return
        if not isinstance(original, dict):
            return
        for key, value in list(chunk.items()):
            if str(value).strip():
                continue
            src = original.get(key)
            if src is None or not str(src).strip():
                continue
            logger.warning(f"段 {key} 译文为空，已退回原文: {str(src)[:40]!r}")
            chunk[key] = str(src)

    def _result_handler(self, result: str, origin_prompt: str, logger):
        try:
            out = super()._result_handler(result, origin_prompt, logger)
        except PartialAgentResultError as e:
            self._restore_blanks(e.partial_result, origin_prompt, logger)
            raise
        if isinstance(out, dict):
            self._restore_blanks(out, origin_prompt, logger)
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
        ids = ids_of(prompt)
        if ids is None:
            return headers, data
        # response_format 在父类里是最后写的，这里覆盖它即可（extra_body 已合并完）
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "segments",
                "schema": schema_for(ids),
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
    assert ids_of(p) == ["3", "4"], ids_of(p)

    s = schema_for(["3", "4"])
    assert s["required"] == ["3", "4"]
    assert s["additionalProperties"] is False
    assert set(s["properties"]) == {"3", "4"}

    # 续写 prompt 没有 <input> 块 —— 必须优雅退回，不加 schema
    assert ids_of("你之前的翻译输出被截断了。请继续。") is None
    # 空 chunk 不生成 schema
    assert ids_of(generate_prompt("{}", "简体中文")) is None

    # 空串必须退回原文，否则 docx 里的 "II." 会被抹掉
    class _L:
        msgs: list = []

        def warning(self, m):
            _L.msgs.append(m)

    agent = SchemaSegmentsAgent.__new__(SchemaSegmentsAgent)
    p2 = generate_prompt(
        json.dumps({"1": "II.", "2": "hello"}, ensure_ascii=False), "简体中文"
    )
    chunk2 = {"1": "", "2": "你好"}
    agent._restore_blanks(chunk2, p2, _L())
    assert chunk2 == {"1": "II.", "2": "你好"}, chunk2
    assert len(_L.msgs) == 1, _L.msgs
    # 原文本身就是空的，不该报 warning
    _L.msgs.clear()
    p3 = generate_prompt(json.dumps({"1": ""}, ensure_ascii=False), "简体中文")
    chunk3 = {"1": ""}
    agent._restore_blanks(chunk3, p3, _L())
    assert chunk3 == {"1": ""} and not _L.msgs

    print("schema_agent 自检通过")


if __name__ == "__main__":
    demo()
