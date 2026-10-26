# sm70-attn 性能盘点（2026-10-26，当前构建）

本文只收**实测**数字，每行都带来源。同一张表内才是可比数字；跨表（单卡 vs TP2、
不同权重、不同 ub）不要直接相减。原始数据：`bench/perf-2026-10-26-tp2.csv`。

## A. 生产配置（当前最优）：2×V100 TP2 + GSQ-RCO-IQ3_S-mtp

```
CUDA_VISIBLE_DEVICES=0,1
LLAMA_SM70_D256=1 LLAMA_SM70_D256_DECODE=1 LLAMA_SM70_D256_DEC_Q4=1
LLAMA_SM70_SKINNY=2 LLAMA_SM70_IQ_SKINNY=1
llama-server -sm tensor -ngl 99 -fit off -c 229376 -b 4096 -ub 8192
  -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3
  -t 12 -tb 16 --temp 0 -n 320 -np 1
```

同一 server 进程内顺序发 4 个 prompt（`cache_prompt=false`，`n_predict=96`，
temp 0），`LLAMA_SM70_D256` 1/0 各跑一遍全序列（2026-10-26 一次会话，
`sweep_perf.csv`，16:35 完成）：

| prompt tokens | prefill tuned | prefill stock | prefill 增益 | decode tuned | decode stock |
|---:|---:|---:|---:|---:|---:|
| 22,081 | **1554.1** | 1404.3 | +10.7% | 72.8 | 66.8 |
| 46,300 | **1435.1** | 1210.0 | +18.6% | 53.3 | 77.6 |
| 100,126 | **1128.5** | 875.0 | +29.0% | 47.7 | 59.2 |
| 174,479 | **869.5** | 638.3 | **+36.2%** | **46.9** | 36.5 |

- 小 prompt 的固定开销很大：554-token 预填充只有 797–860 tok/s（`final_tp2.log`），
  勿与长 prompt 画在同一条曲线上。
- 同构建其它运行的**观测最好值**：prefill 1612 @22k（`final_tp2.log` 15:27）、
  878 @174k（同文件）——与上表差 1–3%，属正常波动；画图建议用上表（一次会话内
  测得，口径一致）。
- **decode 列不要当曲线画**：MTP 投机接受率随生成文本变化，同一上下文不同 prompt
  能差 ±15%（stock 46k 的 77.6 就高于 22k 的 66.8）。只有同一 prompt 的 tuned/stock
  对比有意义：174k 上 **46.9 vs 36.5 = +28.5%**。
- 增益随上下文单调上升（+10.7% → +36.2%），因为 D256 kernel 的价值随注意力占比增长；
  注意力在 174k 预填充中约占 61%（见 `docs/sm70-attn-d256-prefill-limit.md`）。

## B. 单卡参考（同一构建、同一会话，`CUDA_VISIBLE_DEVICES=1`）

配置与 A 相同，但去掉 `-sm tensor`、并且 **不设** `LLAMA_SM70_SKINNY` /
`LLAMA_SM70_IQ_SKINNY`：`SKINNY=2`（第二份量化权重）在单张 32GB 卡
`-c 229376` 加载即 OOM；`IQ_SKINNY=1` 单独开占用 32,439 MiB / 32,768 MiB，
没有余量。原始数据 `bench/perf-2026-10-26-1gpu.csv`，图 `sm70-attn-perf-1gpu.png`。

| prompt tokens | prefill tuned | prefill stock | 增益 | decode tuned | decode stock |
|---:|---:|---:|---:|---:|---:|
| 22,081 | **1022.3** | 923.4 | +10.7% | 50.2 | 54.4 |
| 46,300 | **921.3** | 765.8 | +20.3% | 48.5 | 38.1 |
| 100,126 | **735.9** | 549.3 | +34.0% | 39.2 | 30.9 |
| 174,479 | **583.4** | 393.8 | **+48.1%** | **26.0** | 19.6 |

- TP2 相对单卡（tuned 对 tuned）：prefill **+52% @22k、+49% @174k**。
- 单卡 decode 的 22k 点是噪声（−7.9%，贪心文本在 tuned/stock 间分叉导致 MTP
  接受率不同）；46k 起 KV 切分解码 kernel 的收益才稳定（+27% / +32%）。
- 早期记录里的单卡 1200.6 @22k / 661.7 @174k 是开着 skinny 类开关的另一套配置，
  与上表不可混用。

## C. 历史头牌（按时间，用于"进展"图；配置不同，勿与 A 混算）

| 日期 | 场景 | 值 | 备注 / 来源 |
|---|---|---|---|
| 8/26 | 176k prefill，单卡 | stock 372.9 → tuned **521.9**（+39.9%） | README 唯一同日 A/B，Q4_K_XL |
| 8/24 | 176k prefill，单卡 | q4-direct 457 vs mirror **474** | 故 q4-direct 默认 OFF |
| 10/25 | 174k decode 阶梯 | 15.1 → 24.0 → 34.99 → 37.37 | 单卡 → 单卡 spec → TP2 GSQ MTP3 → +p2p AR |
| 10/25 | 174k decode | **43.5**（48 splits，TC decode kernel） | `tp2_sp48_deep.log` |
| 10/25 | 30k decode | **90.6**（ub8192 + p2p AR） | `tcc-tp2-findings.md` |
| 10/26 | TP2 prefill @22k | 739 → **1558–1612** | p2p AR 修复（4 个 commit） |
| 10/26 | TP2 prefill @174k | 502 → **869–878** | 同上 |
| 10/26 | TP2 prefill @22k | 1980（**仅计时**，跳过 allreduce） | `tp2_arskip.log`，AR 已是 3.4% |

## D. 内核级（画"效率"Bar 图用）

| 项 | 值 | 来源 |
|---|---:|---|
| D256 prefill kernel @174k（单层） | 357.7 ms = 48.4 TFLOP/s = **38.7%** 峰值 | `verify/sm70_attn_micro.cu` |
| 其中 MMA / 胶水（零重叠） | 133 ms / 225 ms | 消融，同上 |
| dense（无 SplitKV3） | 370.0 ms（SplitKV3 +3.4%） | 同上 |
| q4-direct K/V（无 f16 mirror） | 507.0 ms（**慢 42%**） | 同上 |
| V100 HMMA.884 真实峰值 | 4 累加器链 38% / 8 链 97%（121.6 TFLOP/s） | `verify/sm70_mma_peak.cu` |

## 画图

```bash
# TP2（生产）
python bench/plot_sm70_perf.py                       # -> bench/sm70-attn-perf.html
bench\shot_chart.bat bench\sm70-attn-perf.html sm70-attn-perf.png

# 单卡（参考）
python bench/plot_sm70_perf.py --csv bench/perf-2026-10-26-1gpu.csv     --out bench/sm70-attn-perf-1gpu.html --ymax 1200 --ystep 400     --sub "单张 V100-SXM2 · Qwen3.8-27B GSQ-RCO-IQ3_S-mtp · -fa on, q4_0 KV · tok/s vs prompt length"     --note "实测 2026-10-26 · 单卡默认 knobs（SKINNY=2 在 -c 229k 单卡 OOM）· n_predict=96 · decode 受 MTP 接受率影响（±15%），仅同 prompt 对比有效 · 数据 bench/perf-2026-10-26-1gpu.csv"
bench\shot_chart.bat bench\sm70-attn-perf-1gpu.html sm70-attn-perf-1gpu.png
```

生成器零第三方依赖（这台机器装不上 matplotlib）：Python 直接输出内联 SVG 的
HTML，再由 headless Edge/Chrome 截成 1400×940 PNG。换数据只需改 CSV 或用
`--csv/--out/--sub/--note/--ymax/--ystep` 覆盖。x 轴用实测 token 数，不插值。
