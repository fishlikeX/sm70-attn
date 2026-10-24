#!/usr/bin/env bash
# P2 + q4 布局修复的全套真机复验 runbook（V100 空闲后一键执行）。
# 顺序即依赖序：先单元，再整模型正确性，最后性能与默认开关决策数据。
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

echo "== [1/6] decode 内核独立测试器（13 case，q4 规范布局）=="
bash verify/run_dec.sh || exit 1

echo "== [2/6] prefill 电池（q4 系 case 以修正后的 CPU 参照重跑）=="
bash verify/run.sh || exit 1

echo "== [3/6] prefill q4-direct 服务器门（D-vs-S 逐位一致，修正后才是有效门）=="
bash verify/logit_q4direct.sh || echo "!! q4direct 门失败——检查 docs/q4-layout-audit.md 的遗留项"

echo "== [4/6] decode logit A/B（P2 规范版 vs stock，两档）=="
bash ~/zcode_work/decode_ab.sh
bash ~/zcode_work/decode_ab_deep.sh

echo "== [5/6] decode 性能 A/B（深档翻盘复测）=="
bash ~/zcode_work/p2_ab.sh

echo "== [6/6] 决策数据齐备：零分叉 + 深档 >= +40% => 建议 DEC/DEC_Q4 转默认开 =="
