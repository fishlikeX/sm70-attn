#!/usr/bin/env bash
# logit_q4direct.sh — q4-direct 数值验证（三相: direct vs staged vs stock）
# 判据:
#   D(direct) vs S(staged) : 逐位一致（核内 dequant 与 to_fp16 staging
#                             舍入逐位等价的直接证据）
#   D vs O(stock)          : 已知 f16 地板内的合法分歧（参考值）
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q3_K_M-fixed.gguf
PROMPT_FILE=verify/dump_prompt3k.txt   # 3k tokens: q chunks >=256, kv >=2048 -> splitkv3

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

start_server() { # $1 = tag, $2/$3 = env kv pairs appended
  local tag=$1; shift
  echo "--- server [$tag] ($*) ..."
  env CUDA_VISIBLE_DEVICES=2 "$@" "$BIN" -m "$MODEL" --mmproj "$MMPROJ" \
    --main-gpu 0 -c 8192 -b 4096 -ub 512 -ngl 99 -fa on -ctk q4_0 -ctv f16 \
    --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
    -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 256 -np 1 \
    --host 127.0.0.1 --port 8080 > /tmp/q4dir_${tag}_server.log 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 200); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && { echo "    ready"; return 0; }
    kill -0 "$SRV_PID" 2>/dev/null || { tail -15 /tmp/q4dir_${tag}_server.log; exit 1; }
    sleep 2
  done
  echo "ERROR: server [$tag] not ready"; exit 1
}

grab() { # $1 = tag
  local tag=$1
  python3 - "$tag" <<'EOF'
import json, sys, urllib.request
tag = sys.argv[1]
prompt = open("verify/dump_prompt3k.txt", encoding="utf-8").read()
body = json.dumps({"prompt": prompt, "n_predict": 128, "temperature": 0,
                   "top_p": 1, "top_k": 1, "n_probs": 1}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                             headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
toks = []
for p in d.get("completion_probabilities", []):
    t = p[0] if isinstance(p, list) else p
    toks.append({"tok": t.get("content"), "p": t.get("prob")})
json.dump(toks, open(f"/tmp/q4dir_{tag}.json", "w"))
print(f"    [{tag}] {len(toks)} tokens saved")
EOF
}

run_phase() { # $1 = tag, rest = env
  local tag=$1; shift
  start_server "$tag" "$@"
  grab "$tag"
  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
  sleep 5
}

run_phase direct LLAMA_SM70_D256_Q4_DIRECT=1
run_phase staged
run_phase stock  LLAMA_SM70_D256=0

python3 - <<'EOF'
import json
def load(t): return json.load(open(f"/tmp/q4dir_{t}.json"))
d, s, o = load("direct"), load("staged"), load("stock")
def cmp(a, b, la, lb):
    n = min(len(a), len(b)); worst = 0.0; fork = None
    for i in range(n):
        if a[i]["tok"] != b[i]["tok"] and fork is None: fork = i
        if a[i]["p"] is not None and b[i]["p"] is not None:
            worst = max(worst, abs(a[i]["p"] - b[i]["p"]))
    print(f"{la} vs {lb}: len {len(a)}/{len(b)}  max|dp|={worst:.3e}  "
          f"first tok fork: {fork if fork is not None else 'NONE'}")
cmp(d, s, "direct", "staged")
cmp(d, o, "direct", "stock")
EOF
