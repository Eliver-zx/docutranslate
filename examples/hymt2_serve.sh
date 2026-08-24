#!/usr/bin/env bash
# Hy-MT2-30B-A3B(int4) vLLM 服务启停。放到模型所在机器上执行。
#
#   ./hymt2_serve.sh start|stop|restart|status|logs
#
# 参数取自实测（30 篇 / 77.7 万字符冷启动）：
#   4096/16  243s 599出tok/s      4096/64  177s 831出tok/s KV81%
#   8192/32  191s 765出tok/s KV41%  ← 采用：比最快慢 8%，KV 留 59% 余量
# 详见 batch.toml.example 的 [profile.hymt2] 注释。

set -uo pipefail

MODEL_DIR=${MODEL_DIR:-/data_big/eliver/hy-mt2-30b-int4}
VLLM=${VLLM:-/data_big/eliver/qwen3.5-9B/.venv/bin/vllm}
PYTHON=${PYTHON:-/data_big/eliver/qwen3.5-9B/.venv/bin/python3}
GPU=${GPU:-6}
PORT=${PORT:-8003}
NAME=${NAME:-Hy-MT2-30B}
MAX_LEN=${MAX_LEN:-8192}   # 不可降到 4096：最长段 3418 tok + max_tokens 1536 会撞 400
MAX_SEQS=${MAX_SEQS:-32}
MEM_UTIL=${MEM_UTIL:-0.92}

LOG="$MODEL_DIR/serve.log"
PIDFILE="$MODEL_DIR/serve.pid"

# 取正在跑的服务 pid。不用 pkill -f：ssh 执行时那个模式会匹配到承载命令的
# shell 自己，pkill 把自身进程组杀掉，后面的启动语句根本不执行。
running_pid() {
    [[ -f $PIDFILE ]] || return 1
    local pid
    pid=$(<"$PIDFILE")
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    rm -f "$PIDFILE"
    return 1
}

# Intel 那份权重的 config.json 带一条写错的正则：想保护 MoE router 却匹配不到
# （router 真名 mlp.router.gate.weight），反而误伤 shared expert 的 gate_proj，
# 加载时报 "Fused module ... requires consistent quant config"。权重里所有
# linear 本就统一 4bit，router 本就是未量化 fp16，这条删掉即可。幂等。
fix_config() {
    "$PYTHON" - "$MODEL_DIR/config.json" <<'PY'
import json, shutil, sys
p = sys.argv[1]
d = json.load(open(p))
qc = d.get("quantization_config", {})
if "extra_config" in qc:
    shutil.copyfile(p, p + ".orig")
    qc.pop("extra_config")
    json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
    print("config.json: 已删除 extra_config（原文件存为 config.json.orig）")
PY
}

start() {
    if pid=$(running_pid); then
        echo "已在运行 (pid $pid)"
        return 0
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "端口 $PORT 被别的进程占用，先腾出来" >&2
        return 1
    fi
    fix_config || return 1

    # VLLM_USE_FLASHINFER_SAMPLER=0 必带：无 nvcc 时 FlashInfer 的采样内核要
    # JIT 编译，报 "Could not find nvcc"。错误出在 sampler 不是 torch.compile，
    # --enforce-eager 治不了。关掉后 torch.compile 与 CUDA Graph 均正常。
    CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 \
        nohup "$VLLM" serve "$MODEL_DIR" \
        --served-model-name "$NAME" --host 0.0.0.0 --port "$PORT" \
        --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
        --gpu-memory-utilization "$MEM_UTIL" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "启动中 (pid $(<"$PIDFILE"), GPU$GPU, $MAX_LEN/$MAX_SEQS)，日志 $LOG"

    for _ in $(seq 60); do
        if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" || true
            echo "就绪: http://127.0.0.1:$PORT/v1  模型 $NAME"
            return 0
        fi
        if ! running_pid >/dev/null; then
            echo "进程已退出，日志末尾：" >&2
            tail -20 "$LOG" >&2
            return 1
        fi
        sleep 5
    done
    echo "5 分钟未就绪，自己看日志：tail -f $LOG" >&2
    return 1
}

stop() {
    if ! pid=$(running_pid); then
        echo "未在运行"
        return 0
    fi
    kill "$pid"
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || { rm -f "$PIDFILE"; echo "已停止"; return 0; }
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null
    rm -f "$PIDFILE"
    echo "已强制停止"
}

status() {
    if pid=$(running_pid); then
        echo "运行中 (pid $pid)"
    else
        echo "未运行"
        return 1
    fi
    curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
        && echo "接口正常: http://127.0.0.1:$PORT/v1" \
        || echo "进程在但接口不通（可能仍在加载）"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
        | sed -n "$((GPU + 1))p" | sed 's/^/显存: /'
    grep -oE 'Running: [0-9]+ reqs, Waiting: [0-9]+ reqs.*GPU KV cache usage: [0-9.]+%' "$LOG" 2>/dev/null \
        | tail -3
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 3; start ;;
    status)  status ;;
    logs)    tail -f "$LOG" ;;
    *)       echo "用法: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
