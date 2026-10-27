<div align="center">

# sm70-attn

**为 Tesla V100 重新点亮的 FlashAttention —— llama.cpp 深度定制分叉**

针对 Qwen3.5 / 3.6 / 3.8-27B（head_dim=256、GQA 6:1、16 层全注意力）的
SM 7.0 CUDA kernel 插件；在 NVLink 上扩展到 2×V100 张量并行，带 D256
张量核心投机解码，以及一套能抓住回归的确定性体系。

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![SM](https://img.shields.io/badge/SM_7.0-Volta-76B900?logo=nvidia&logoColor=white)
![FA](https://img.shields.io/badge/FlashAttention-D256%20Split--D-3b82f6)
![SplitKV3](https://img.shields.io/badge/KV%20Split-3--way-22c55e)
![TP2](https://img.shields.io/badge/2%C3%97V100-TP2%20%2B%20P2P%20NVLink-8957e5)
![Perf](https://img.shields.io/badge/174k%20prefill-%2B36.2%25-16a34a)
![Decode](https://img.shields.io/badge/174k%20decode-46.9%20tok%2Fs-22c55e)
![Gates](https://img.shields.io/badge/harness-23%2F23%20PASS-6366f1)

[English](README-sm70.md) · **中文**

</div>

---

<img src="sm70-attn-perf.png" alt="sm70-attn vs stock（2×V100 TP2）：预填充吞吐 vs prompt 长度">

## 为什么存在这个分叉

llama.cpp 官方 flash-attention 路径对 SM 7.0（Volta / Tesla V100）没有
tensor-core 实现。本仓库把一颗经过 1CatAI 验证的 Split-D N32 D256 kernel
移植进 llama.cpp 主线，并在其上搭出生产栈：

**预填充**
- Split-D D256 预填充 kernel（移植本体）+ **SplitKV3**：长前缀 chunk 的三路
  KV 切分（174k 实测 +3.4%）；
- **注意力输出全链路 f32**：不再经 f16 暂存，单层舍入减半；
- **q4_0 核内直读**（opt-in）：XQA 风格的按 dtype 模板加载，把原始 block
  直接解到 MMA 寄存器 fragment，省掉 f16 mirror。
- **q8_0 核内直读**（opt-in，PR #3）：同一套模板架构支持 q8_0 K/V，与 staged
  f16 mirror **逐位一致**（本机 22k 与 174k 实测，top-3 logprob 到 9 位小数
  完全相同）。默认关闭：本机 mirror 只占 0.94 GB/卡（229k），而 direct 在 174k
  慢约 9%；它是给镜像放不下的权重组合准备的。

**解码**
- **D256 张量核心解码 kernel**：面向投机验证宽度（w≥2），带核内 q4_0 载入和
  二分查找活键上界；
- **自适应 KV 切分**（`kv_len/920`，上限 48）：深上下文解码不再把 SM 网格晾着；
- **IQ 瘦身重排**（opt-in）：174k 解码 +8.7%、预填充 +7%。

**GDN（线性注意力）**
- **l2norm prologue 融合**（opt-in，逐位精确）：把每层 q/k 的 RMS-norm + scale
  链折进 GDN kernel，一趟解码省掉 4 个节点 × 48 层。逐位一致是构造出来的
  （每 lane 一个 float4 + 与 `rms_norm_f32` 相同的 32 宽 shuffle 蝶形）：
  16,384 个元素里 0 个不同。**在测过的工况下没有轮次收益**（每轮墙钟中位数
  相同；tok/s 的 A/B 会被草稿接受率波动混淆）——作为负性能结果归档、默认
  关闭——但这项工作找出了那条从未被调用的融合归约里 4× 过计的 bug 并修掉了。

**多卡**
- **2×V100 张量并行（NVLink，Windows TCC）**，含面向预填充级消息的单发
  **P2P allreduce**——修好之后在 22k 上与"完全不做 allreduce"落在噪声内
  （739 → 1558–1612 tok/s）。

**正确性**
- **sm_70 去量化 kernel**（iq3_s / iq3_xxs / iq4_xs）：与 stock 去量化路径
  **逐位一致**，默认开启；
- 定位并修复两个**不确定性根因**（去量化 LUT 越界、填充后的因果基准），
  每个都配了能在旧代码上失败的 gate；
- mtmd + DFlash2 崩溃修复（上游 ggml-org#27408），以及服务重启后的上下文
  检查点持久化。

所有行为都能用一个环境变量整条回滚到 stock 路径，且逐位一致。

## 性能

测试机：2×Tesla V100-SXM2-32GB（NVLink、TCC）+ 一张 RTX 3080 供桌面使用。
权重 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp`。下面每个数字都是实测；原始数据和画图
脚本在 `bench/`，完整出处见 `docs/sm70-perf-inventory.md`。

### 生产配置：2×V100 TP2（当前最优）

```
CUDA_VISIBLE_DEVICES=0,1
LLAMA_SM70_D256=1 LLAMA_SM70_D256_DECODE=1 LLAMA_SM70_D256_DEC_Q4=1
LLAMA_SM70_SKINNY=2 LLAMA_SM70_IQ_SKINNY=1
llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  -sm tensor -ngl 99 -fit off -c 229376 -b 4096 -ub 8192 \
  -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3 \
  -t 12 -tb 16 --temp 0 -np 1
```

同一个 server 进程内顺序发 4 个 prompt（`cache_prompt=false`、`n_predict=96`、
temp 0），tuned 与 `LLAMA_SM70_D256=0`（stock）各跑一遍：

| prompt tokens | 预填充 sm70-attn | 预填充 stock | 预填充增益 | 解码 sm70-attn | 解码 stock |
|---:|---:|---:|---:|---:|---:|
| 22,081 | **1554.1** | 1404.3 | +10.7% | 72.8 | 66.8 |
| 46,300 | **1435.1** | 1210.0 | +18.6% | 53.3 | 77.6 |
| 100,126 | **1128.5** | 875.0 | +29.0% | 47.7 | 59.2 |
| 174,479 | **869.5** | 638.3 | **+36.2%** | **46.9** | 36.5 |

- 同构建其它运行的观测最好值：**1612 tok/s @22k**、**878 tok/s @174k**
  （1–3% 波动）。
- 增益随上下文增长，因为注意力随上下文增长：174k 时注意力约占预填充墙钟的
  **61%**，22k 时约 14%。
- **解码列不要跨 prompt 连成曲线**：MTP 接受率取决于生成文本（±15%）。只有
  同一 prompt 的成对比较有意义：174k 上 **46.9 vs 36.5 = +28.5%**。
- P2P 修复后 allreduce 的成本：完全跳过它是 1591 tok/s @22k，带它是
  1558–1612，落在噪声内；修复前它占整趟的 62%。

### 单卡（同构建，参考）

<img src="sm70-attn-perf-1gpu.png" alt="sm70-attn vs stock（单张 V100）：预填充吞吐 vs prompt 长度">

| prompt tokens | 预填充 sm70-attn | 预填充 stock | 预填充增益 | 解码 sm70-attn | 解码 stock |
|---:|---:|---:|---:|---:|---:|
| 22,081 | **1022.3** | 923.4 | +10.7% | 50.2 | 54.4 |
| 46,300 | **921.3** | 765.8 | +20.3% | 48.5 | 38.1 |
| 100,126 | **735.9** | 549.3 | +34.0% | 39.2 | 30.9 |
| 174,479 | **583.4** | 393.8 | **+48.1%** | **26.0** | 19.6 |

单卡系列跑的是**默认** skinny 开关：TP2 配方的 `LLAMA_SM70_SKINNY=2`
（量化权重再来一整份拷贝）在单张 32GB 卡上 `-c 229376` 直接加载 OOM；
`IQ_SKINNY=1` 单独开也几乎顶满（32,439 MiB / 32,768 MiB）。其余参数与 TP2 配方一致。
TP2 相对单卡：预填充 **+52% @22k、+49% @174k**。

单卡解码也遵循 TP2 的注意事项：22k 的 tuned/stock 对在 MTP 接受率噪声内
（−7.9%，贪心文本已经分叉），但从 46k 起 KV 切分解码 kernel 明显起作用
（46k/100k +27%，174k +32%）。

### 并发优先变体：同样两张卡上 4 × 256k

两个瘦身布局（`LLAMA_SM70_SKINNY=2` + `LLAMA_SM70_IQ_SKINNY=1`）**只买预填充**：
关掉后同一组 prompt 实测 22k 1554 → 1498 t/s（−3.6%）、174k 877 → 841 t/s
（−4%），而**解码不变**（关掉投机的 A/B，排除草稿接受率干扰：174k 23.4 vs
23.3 t/s，22k 40.5 vs 40.1）。此前记在 IQ 重排头上的"+8.7% 解码"是接受率
效应——重排会让 logits 有极小差异，MTP 验证的草稿数随之不同。

它们真正花的是显存：同一个 22k 实例里每卡 22,821 → 16,904 MiB（−5.9 GB），
其中 4.9 GB 是权重拷贝本身（`-c 8192` 固定开销 13,075 → 8,151 MiB）。这个
量比第 4 路 256k 需要的 4.15 GB/卡还多，所以当并发比最后 4% 预填充更重要时
就该用这套：

| 配置 | 预填充 22k | 预填充 174k | 能塞几个 256k 槽 |
|---|---:|---:|---|
| 生产（瘦身全开） | 1554 t/s | 877 t/s | 3（31,126 MiB/卡；4 路 OOM） |
| 并发优先（瘦身全关） | 1498 t/s | 841 t/s | **4（30,314 MiB/卡）** |

```bash
# 4 × 256k：两个瘦身开关都保持默认（0）
CUDA_VISIBLE_DEVICES=0,1 \
LLAMA_SM70_D256=1 LLAMA_SM70_D256_DECODE=1 LLAMA_SM70_D256_DEC_Q4=1 \
./build/bin/llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
    -sm tensor -ngl 99 -fit off -c 1048576 -b 4096 -ub 8192 \
    -fa on -ctk q4_0 -ctv q4_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 -np 4
```

两个容易踩的细节：`LLAMA_SM70_IQ_SKINNY=1` 单独开是**无效**的，IQ 重排的前提是
`LLAMA_SM70_SKINNY != 0`（`iq-skinny.cu`）；中间档 `SKINNY=1 + IQ_SKINNY=1`
（k-quant 原地重排、不复制第二份）实测 20,722 MiB/卡，174k 854 t/s。

### 8/26 的单卡 A/B（历史）

上一版 README 的头牌数字来自另一套栈（Q4_K_XL、`-ctv f16`、`-ub 512`、无 TP2）：
176k 预填充 **372.94 → 521.93 tok/s（+39.9%）**，同一二进制、同一 prompt，
`LLAMA_SM70_D256=0` vs 默认。它作为后续工作的基准保留，与上表不可直接比较。

### 剩下的时间花在哪

D256 预填充 kernel 在 174k 生产形状下跑到 **48.4 TFLOP/s = V100 125 TFLOP/s
峰值的 38.7%**。消融证明瓶颈不在 MMA：133 ms 的张量核心工作 + 225 ms 完全不
重叠的 staging/softmax 胶水（合计 358 ms）。裸 MMA 探针解释了 38.7%：每个 warp
需要约 8 条独立累加器链才能到 97% 峰值，而 QK gemm 只有 4 条。现有指令流下的
天花板是 **225 ms（1.59×）**；要够到它得改成双块 ping-pong 调度，而不是再调
一个常数。完整证据：`docs/sm70-attn-d256-prefill-limit.md`。

## 快速开始

```bash
# 编译（需要 CUDA >= 12.x，目标 GPU 为 Volta）
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j

# 2×V100 张量并行（生产）。Windows/TCC 下：
#   - CUDA 枚举顺序把两张 V100 排在最前 -> CUDA_VISIBLE_DEVICES=0,1
#   - 必须 -fit off，否则加载器会试图把一层放到 CPU 上
#   - 需要 NVLink 在位（见下文「TP2 on 2×V100」）
CUDA_VISIBLE_DEVICES=0,1 \
LLAMA_SM70_D256=1 LLAMA_SM70_D256_DECODE=1 LLAMA_SM70_D256_DEC_Q4=1 \
LLAMA_SM70_SKINNY=2 LLAMA_SM70_IQ_SKINNY=1 \
./build/bin/llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
    -sm tensor -ngl 99 -fit off -c 229376 -b 4096 -ub 8192 \
    -fa on -ctk q4_0 -ctv q4_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 127.0.0.1 --port 8080

# 单卡（同参数，去掉 -sm tensor 与两个 skinny 开关）
CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=1 LLAMA_SM70_D256_DECODE=1 \
LLAMA_SM70_D256_DEC_Q4=1 \
./build/bin/llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
    -ngl 99 -fit off -c 229376 -b 4096 -ub 8192 \
    -fa on -ctk q4_0 -ctv q4_0 \
    --spec-type draft-mtp --spec-draft-n-max 3
```

非 Volta / 非 D256 / 非因果 / 小批量（ne[1]<256）会自动回落到 llama.cpp
原生路径，无需配置。

### 走不通的路（都已实测，别盲目重试）

| 组合 | 结果 |
|:---|:---|
| WSL2 + TP2 | 两张 V100 在 GPU-PV 下看不到 NVLink P2P；请在 Windows/TCC 上跑 |
| `--spec-type draft-dflash` × `-sm tensor` | `ggml-backend-meta.cpp` 断言（按行算子拿到 axis-0 切分源）；改用 `draft-mtp` |
| `-sm layer` 做解码 | 解码零收益（同步层链无重叠），只有预填充 +39% |
| `-ub 16384` | OOM；8192 是预填充最优（预填充受每 ubatch 去量化限制） |
| 单卡开 `LLAMA_SM70_SKINNY=2` | OOM（第二份权重拷贝），TP2 专用 |
| 把 `q4-direct` 设默认 | 174k 下 kernel 慢 42%（窄 u16 载入）；只在显存吃紧时开 |
| `-np > 1`（多序列同批） | D256 内核只实现单序列因果（因果基准由最后一行 mask 前缀探测）。现在**自动回落**：per-seq 缓存下混批的 `ne[3] != 1` 会被门禁拒绝；unified 缓存 + >1 槽由服务端置 `LLAMA_SM70_D256_MULTI_SEQ=1`，整进程走 stock。单序列批仍走内核 |
| 含 Q8_0 权重的模型 + `LLAMA_SM70_SKINNY=2` | Q8_0 没有侧缓冲编解码，加载时会断言退出；保持 `SKINNY=0/1` 或换量化 |
| 各种 kernel 编译选项微调 | 已经撞到累加器链上限，见「剩下的时间花在哪」 |

## 环境变量

| 变量 | 默认 | 作用 |
|:---|:---:|:---|
| `LLAMA_SM70_D256` | `1` | 总开关；`0` = stock 路径，逐位一致 |
| `LLAMA_SM70_SPLITKV3_MIN_KV` | `2048` | SplitKV3 启用阈值（kv_len）；`0` 关闭 |
| `LLAMA_SM70_D256_Q4_DIRECT` | `0` | 核内 q4_0 直读（省 f16 K/V mirror）：省显存，预填充慢约 4% |
| `LLAMA_SM70_D256_DECODE` | `0` | D256 张量核心解码 kernel（投机验证宽度 w≥2） |
| `LLAMA_SM70_D256_Q8_DIRECT` | `0` | q8_0 K/V 按原始 block 核内直读。与 staged mirror 逐位一致（22k / 174k 的 top-3 logprob 都相同），但 174k 预填充慢约 9%（792 vs 871 t/s）：本机 f16 mirror 只占 0.94 GB/卡，所以这个 opt-in 是给显存吃紧的场景。TC 解码 kernel 只吃 f16/q4_0，q8_0 缓存的解码走 stock 路径 |
| `LLAMA_SM70_D256_DEC_Q4` | `0` | 解码 kernel 的 q4_0 直读 |
| `LLAMA_SM70_D256_SPLITS` | `0` | 解码 KV 切分上限覆盖（`0` = 自动 `kv_len/920`，上限 48；最大 256） |
| `LLAMA_SM70_DEQUANT_FAST` | `1` | sm_70 iq3_s/iq3_xxs/iq4_xs 去量化 kernel；`0` = stock 去量化 |
| `LLAMA_SM70_SKINNY` | `0` | q4_K 瘦身路径：`1` = 原地重排（不复制），`2` = 多一整份权重拷贝（本机 +1.1 GB/卡）。同时是 `IQ_SKINNY` 的开关前提。实测代价见「并发优先变体」 |
| `LLAMA_SM70_IQ_SKINNY` | `0` | IQ 重排：174k 预填充 +4%、**解码不变**、side buffer +3.8 GB/卡。仅在 `LLAMA_SM70_SKINNY != 0` 时生效 |
| `LLAMA_SM70_D256_DEBUG` | `0` | 打印 kernel 选择探针（ACCEPT/REJECT + 原因） |
| `LLAMA_SM70_D256_MULTI_SEQ` | 由服务端设置 | unified KV + >1 槽时由 llama-server 置 `1`，整进程停用 D256 内核（内核只实现单序列因果）。手动跑其他前端时可自行设置 |
| `LLAMA_SM70_IQ_PARITY` | `0` | `1` = 首次 repack 时把每个类型的一个张量分别用侧缓冲展开和 stock kernel 去量化、逐值比对，打印 `nbad`（10/26 的 IQ3_S 全零权重就是这么定位的） |
| `LLAMA_SM70_SKINNY_DEBUG_STOCK` | `0` | 调试切分：`1` = IQ 类张量强制走原始路径，`2` = k-quant 类走原始路径 |
| `GGML_CUDA_P2P_AR_MAX_BYTES` | `176 MiB` | 单发 P2P allreduce 的消息窗口；`0` 关闭该路径 |
| `GGML_CUDA_P2P_AR_NBLOCKS` | `80` | allreduce block 数（≤ 每 SM 一个，超了会死锁） |
| `LLAMA_SM70_GDN_PROLOGUE` | `0` | GDN l2norm prologue 融合（逐位精确；`_DBG=1` 匹配诊断，`_NOCACHE=1` 关 cpy 折叠） |
| `LLAMA_SM70_GDN_CHUNK` | `1` | GDN 分块（预填充）路径；`0` 关闭 |
| `LLAMA_SM70_STEP_PROF` | `0` | 逐轮/子图墙钟账本（P0，`ggml-backend-meta.cpp`）；graphs-on 下可用 |
| `SM70_DUMP` / `SM70_DUMP_KV` | 关 | 抓取 kernel 输入/输出，供离线复现与三方归因 |

## 验证体系

每个改动都有 gate 守着，全部自包含、可重跑（`bash verify/<script>`，
二进制用 `nvcc -arch=sm_70` 编译）：

| Gate | 内容 | 状态 |
|:---|:---|:---:|
| Kernel harness | 23 例：dense / f32-out / 尖峰 / 位置优先 K,V / SplitKV3 边界 / q4-direct | **23/23** |
| 去量化 parity | 真实 launch、投毒输出、每次调用都查错：iq3_s/iq3_xxs/iq4_xs 与 stock 逐位一致 | **PASS** |
| 确定性 | 相同请求必须给出逐位相同的末 token logprob（554 / 22k / 174k，单卡与 TP2） | **PASS** |
| logit A/B | 贪心 128 token 解码，与 stock 零分歧 | **PASS** |
| 三方归因 | 首层 ON-OFF 差必须落在 f16 舍入地板（dense ~5e-4） | **PASS** |
| GDN prologue 逐位 | 融合 vs stock `rms_norm+scale`，16,384 元素 | **0 个不同** |
| 多模态 | 图片请求 200 + 描述正确 + 文本接受率 0.88 | **PASS** |
| 回滚 | `LLAMA_SM70_D256=0` 与 stock 逐位一致 | **PASS** |

去量化 parity gate 值得单独说：旧版本比较的是「内核出错后仍保持全零」的 host
缓冲，于是对一个越界读 LUT 的 kernel 也打印 "BIT-EXACT"。现在的 gate 检查每次
launch/拷贝、投毒输出、强制 fp16 scale 有限；它在旧代码上明确失败，在新代码上
通过。

## 修复档案（完整因果链）

### 上下文污染（8/23，9 行修复）

**症状**：23 个「失忆」签名——模型能接上最近的上下文，但更早的内容全忘。

**根因**：launcher 假设 `to_fp16_nc` 的去量化输出是 `[ne2][ne1][ne0]`
（头优先），实际线性化成 `[ne1][ne2][ne0]` = 位置优先 `[ctx][hkv][D]`。
头优先的 stride 把每次 q4_0-K 预填充的 K 都打乱了：每个全注意力层输出约 0.69
相对误差，累积成 O(1) 的 nat logit 漂移和贪心 argmax 翻转。

**诊断**：`sm70_variant_scan.py` 12 分钟锁定（K 转置变体把 |v-C| 从 0.69
降到 0.08）。**修复**：9 行；`posK/posKV` 用例永久钉在 harness 里。

### 多模态 + 投机解码崩溃（8/24，上游 #27408）

**症状**：`--spec-type draft-dflash` 下喂图片，请求崩在
`llama_decode(ctx_dft) rc=-1` → HTTP 500 "failed to process mtmd chunk"。

**根因**：多模态 batch 的每一行位置是常数（752 行图片全在 pos 53，空间结构在
target 的 M-RoPE 里），而后续文本从 `image_pos + grid_height` 继续。草稿模型
的一维 KV cache 两种形状都存不下。

**修复**（`common/speculative.cpp` 三处）：`process()` 跳过 embedding batch，
在下一个 token batch 到达时用零特征行补洞；`draft()` 把噪声块放在草稿缓存自己
的 `pos_max + 1`；把 pos_max-vs-N 的警告降级为调试启发式。target 仍然验证每个
草稿 token，输出分布保持不变。

### IQ3_S 权重全零（10/26）——跨 TU 设备符号

**症状**：`SKINNY=2 + IQ_SKINNY=1` 下，超过约 128 token 的提示被当成"用户没说话"：
模型复述上一轮答案、声称没看到附件、把别处的回答搬过来。短对话完全正常，所以
一直没人怀疑到权重上。

**根因**：IQ3_S 的 design-B 解码要读一张 f16 网格表 `iq3s_grid_f16`。这张表在
`iq-skinny.cu` 里定义并初始化，但消费它的展开 kernel 实例化在 `q4k-skinny.cu`。
本树 CUDA 构建**没有开 relocatable device code**（无 `-rdc`），`extern __device__`
在另一个 TU 里绑不到定义——那个 TU 读到的是一张全零表。于是每个 IQ3_S 权重都
展开成 0.0，模型（FFN 主体正是 144 个 IQ3_S 张量）只剩 system 提示可看。
M≤64 的旁路用的是原始字节，所以短提示不受影响，只有长提示会炸。

**诊断**：`LLAMA_SM70_IQ_PARITY=1` 把同一张量分别用侧缓冲展开和 stock kernel
去量化、在 host 上逐值比对——`iq3_s nbad = 52,420,257 / 52,428,800`（全部为零）；
再配合 `LLAMA_SM70_SKINNY_DEBUG_STOCK` 三向切分（只把 IQ 类张量强制回原始路径
即恢复正常，强制 k-quant 类则不然）定位到 IQ 编解码。

**修复**：表改成 header 里的内部链接（每个 TU 各持一份），由启动解码 kernel 的
TU 在首次使用前初始化自己那份（`6cb842973`）。修复后 `nbad = 0`；137 / 408 /
1201 token 的哨兵检索全部通过，2.4 KB 附件的问答和追问都从文件内容作答。
修复后复测预填充：IQ_SKINNY 仍然值得开（1k 提示 1480 vs 1433 t/s，8.2k 提示
1891 vs 1858 t/s），代价仍是 +7.3% 显存。

### ≥30k 的不确定性（10/26）——两个独立根因

相同的 llama-server 请求在 22k 以上上下文给出不同输出。两个根因都在 sm70 代码
里，而且都产生**错误数值**，不只是竞态：

1. **IQ4_XS 去量化 LUT 越界**：高半字节读成 `(q >> 8j) >> 4`，没有掩码，于是
   一个 32 位字去索引只有 16 项的查表，远远越界。垃圾值取决于内存里恰好有什么，
   所以一个进程的第一次预填充与后续不同。修复：`& 0xF`（`ac3c27910`）；快速去量化现在在 550
   与 22,081 token 上与 stock 逐位一致。
2. **D256 预填充的因果基准取自填充后的键数**：launcher 传 `kv_len - q_len`，
   但 `mask->ne[0]` 会为分块对齐向上填充，于是首行看到未来 token，末行读到填充
   列的陈旧数据（上一个请求的 KV）。修复：用 mask 末行的活键数做单调二分（`315dbb08e`）——与
   解码 kernel 用的是同一个探针。

同轮还修了旧去量化 gate 的「空转」（见「验证体系」）。新 gate：相同请求在 554 /
22k / 174k、单卡与 TP2 下 logprob 逐位一致。

### TP2 预填充比单卡还慢（10/26）——allreduce 的窗口

症状：张量并行的预填充反而比单卡慢——同一构建、同一 prompt 下 22k 只有
739 tok/s，而单卡是 765（174k 是 502）。完全跳过 allreduce（仅计时）显示它
占整趟的 62%。

**根因**：单发 P2P allreduce 的消息窗口只有 1 MiB。预填充的
`[n_embd × ub]` f32 partial 在 554 token 时是 11 MB、8192 时是 168 MB，于是
根本进不了 NVLink 路径，落进主机中转的 copy-engine 归约：每条消息约 122 ms、
1.4 GB/s 的 PCIe 往返，每层两次。

**修复**（`allreduce-p2p.cu` 四个 commit）：`627961dc0` 把默认窗口抬到
176 MiB（22k 739 → 1472 tok/s，此时归约仍占整趟约 25%）；`597028032` 把
block 数 16 → 80（每 SM 一个；1472 → 1514 → 1544 tok/s）；`4463259d1` 只往
对端暂存并从本地源开始归约（暂存流量减半）；`17cc119fc` 发布改用
`__threadfence()` 替代 `__threadfence_system()`（切片和标志都落在对端设备
内存，不需要系统级刷写）。结果：**22k 739 → 1558–1612 tok/s，174k 502 →
878**；22k 的 AR-skip 对照现在是 1591 vs 1558–1612，已在运行间噪声内。

## 技术深度

### SplitKV3

长前缀 chunk（kv_len ≥ 阈值）把 KV 扫描沿 `gridDim.y` 三路切分，CTA 数 ×3，
让长预填充的后段不再把 KV 扫描串行在已经饱和的网格上；一个小 merge kernel 合并
每行的 (max, sum, 分子) 三元组。今天在 174k 生产形状实测：357.7 ms vs dense
370.0 ms（+3.4%）。上游漏掉的边界：某行的因果窗口可能在一段内被完全遮住，所以
SplitKV3 实例把 `row_max` 初始化为 −1e30（不是 −inf），避免
`(-inf)-(-inf)=NaN` 污染 partial 链；merge 也对分母为 0 做了保护。

### q4-direct：为什么默认 OFF

kernel 有 `Kq4/Vq4` 模板分支，直接读原始 q4_0 block cache
（`[ctx][head][block]`，字节 stride），把值解到寄存器 fragment / smem panel。
舍入与 staged `to_fp16` 路径逐位一致（harness 与 logit A/B 都验过）。174k
实测：q4-direct 让 kernel **慢 42%**（507 vs 358 ms），而它替代的整缓存 f16
暂存只占 770 s 预填充里的约 0.1 s。结论：这个 opt-in 用预填充速度换 f16 mirror
的显存，只在显存吃紧时才值得。

### 解码路径

深上下文解码是延迟问题、不是带宽问题：KV 读只占一步的极小部分，但固定数量的
CTA 会让 SM 网格闲着。三层工作解决了它：D256 张量核心解码 kernel（把 GQA 行
打包给投机验证宽度、q4-direct 载入、二分活键上界）、自适应 KV 切分
（`kv_len/920`，上限 48）、以及 IQ 瘦身重排。174k 解码阶梯：15.1 → 24.0
（单卡 + 投机）→ 34.99（TP2 + GSQ + MTP3）→ 37.37（+P2P allreduce）→
43.5（自适应切分）→ **47.3**（+IQ skinny），上图实测 46.9。

### TP2 on 2×V100（TCC、NVLink）

- NVLink 2.0 ×6 链路：单向 145.1 / 144.9 GB/s，双向聚合 289.6 GB/s
  （理论 300 的 98%）；
- 两张 V100 都开 TCC：P2P 才可靠；Windows 下 CUDA 枚举顺序是
  [V100, V100, 3080]，所以用 `CUDA_VISIBLE_DEVICES=0,1`；
- 必须 `-fit off`，否则加载器会试图把一层放到 CPU；
- TP 是 meta 后端：逐子图派发 + 主机驱动的 allreduce，每层两个归约点（一趟
  预填充 130 个）；
- DFlash 草稿与 TP 不兼容（断言）；`draft-mtp` 可用，是生产投机解码器。

### 上游（1Cat-vLLM）状态：已榨干

- **预填充 kernel**：Split-D 移植 + 修复 + SplitKV3，吃干净了；
- **XQA 解码 kernel（3490 行）**：没有原样移植；D256 张量核心解码 kernel 覆盖了
  我们的验证宽度，上面的阶梯也说明剩余收益在调度而不在 kernel；
- **TurboQuant / FP8 KV / paged layout 转换器**：私有量化格式与 vLLM 专用设施，
  kernel 层不可移植；
- **flash_qla（GDN 线性注意力）**：GDN 层现在做到了逐轮画像（P0 step
  profiler）：融合 prologue 每趟省掉 192 次 launch 却测不出收益，预算在主机编排
  与权重流上（`docs/sm70-gdn-prologue-fusion.md`）；
- **smem bank padding 经验**：我们的 vendor kernel 已经带 pitch-68 +
  Swizzle<3,3,3> + TT-swizzled V 布局。

### 已知限制与下一步杠杆

- 注意力占 174k 预填充墙钟约 61%，而 kernel 在峰值的 38.7%；再往上需要
  `docs/sm70-attn-d256-prefill-limit.md` 里描述的双块 ping-pong 重写。
- ~30k 解码的逐轮预算（P0 step profiler，`docs/sm70-gdn-prologue-fusion.md`）：
  meta 主机编排每趟前向约 3.5–4.0 ms（每轮 55 次子图计算，每次都要过
  `ggml_backend_sched` + CUDA graph launch）；权重流只跑到 roofline 的约 43%
  （每卡每轮约 10.4 ms GPU 忙，900 GB/s 下的理论地板约 3.2 ms）。下一步的杠杆
  是单图双设备捕获与 dequant 融合 TC GEMM，不是继续消除胶水 kernel。

## 工具链

全部自包含，在物理机上直接跑。

`verify/` —— 正确性与微基准：
- `sm70_verify.cu` / `run.sh` —— 23 例 kernel harness，CPU f32 参考，kernel
  源码 md5 钉死；
- `sm70_dequant_parity.cu` —— 真实 launch 的去量化 parity gate（查错、投毒输出）；
- `sm70_attn_micro.cu` —— 生产几何的独立 D256 预填充基准（SplitKV3 / q4-direct /
  dense 可切换，带朴素参考校验）；
- `sm70_mma_peak.cu` —— 裸 `mma.sync.m8n8k4` 峰值 vs 累加器链数；
- `sm70_gdn_prologue_bitexact.cu` / `sm70_gdn_prologue_microtest.cu` —— GDN
  prologue 融合：逐位一致性（0/16384）与 microtest 暴露的 4× 归约过计；
- `ggml/src/ggml-cuda/step-prof.h` + `ggml-step-prof.h` —— P0 轮次/子图墙钟
  账本（环境变量控制，关时零开销）；
- `sm70_gemm_micro.cu` —— 生产 GEMM 形状上 cuBLAS vs 先解量化再 cuBLAS；
- `ablate.py` —— 生成「只计时」的 kernel 变体（对某个内层阶段 `if (false)`），
  不碰生产头文件；
- `sm70_layer_diff.py` / `sm70_variant_scan.py` / `sm70_repro.py` /
  `dump_*.sh` —— 8/23 的三方归因工具包；
- `logit_ab_fixed.sh`、`perf_ab*.sh` —— logit 裁决与吞吐验收。

`tools/sm70gui/` + `tools/webui/` —— 本地 launcher 与 UI：

- Chat / Monitor / About / **Launch** 四个页签，`sm70gui.py` 一个进程同时做静态 UI、
  API 代理（SSE 逐块转发）、GPU/主机遥测和服务生命周期；
- Launch 页签没有"预设档位"：模型、基础参数（上下文/并行槽/KV 类型/投机/线程…）
  直接填，其余命令行参数用一条自由文本追加，环境变量则在完整表格里逐项开关
  （清单与默认值来自源码，`tools/sm70gui/sm70_config.py`）；
- launcher 只自动推导它能**测到**的东西：GGUF 头里的 arch/层数/量化构成/MTP 层，
  以及 CUDA 驱动 API 读到的计算能力（cc 7.0 = SM70）。设备按卡开关，
  `CUDA_VISIBLE_DEVICES` 由开关拼出、只含 SM70 卡；非 SM70 卡不给开关；
- 环境变量分两档：常用 9 个直接列出，其余（TP2/上游/调试）收进折叠的「高级」区
  并注明"非必要不调整"；每条都带英文说明 + 中文说明；
- 界面语言跟随系统（About 里可强制 English / 简体中文），切换即时生效；
- 选择会让某个开关失效时会给出原因（没有 IQ 张量的 `IQ_SKINNY`、没有 MTP 层的投机），
  命令行预览导出的环境与真正 `exec` 的完全一致。

`bench/` —— 负载与测量：
- `prompt_46k.txt`、`prompt_176k.txt` —— 标准 / ROI 预填充负载；
- `perf-2026-10-26-tp2.csv` —— 上表的 TP2 原始数据；
- `plot_sm70_perf.py` + `shot_chart.bat` —— 零第三方依赖的图表生成器
  （HTML+SVG）与无头浏览器截图脚本；
- 跨上下文扫描脚本（`sweep_perf.py` 一类）放在仓库外，因为需要模型和起服务。

## Windows 图形界面（sm70gui）

开箱即用：`tools\sm70gui\sm70gui.exe` **已经打包好放在仓库里**，不需要装 Python、
也不需要自己打包。双击后是它**自己的应用窗口**（内嵌 WebView2），**不会再拉起浏览器
标签页**；网页仍然在 `http://127.0.0.1:8199/`，想用浏览器看就自己输这个地址（托盘
菜单里也有“在浏览器中打开 / Open in browser”）。

同一台电脑只有**一份配置**：对话、采样设置、API key、主题和语言都由 launcher 统一
保存（`sm70gui-ui.json`，和 `sm70gui-config.json` 同目录），窗口端、Edge 或任何其他
浏览器打开同一个地址看到的是同一份，改哪边另一边下次打开/切回来都会同步；模型路径、
参数、环境变量、设备这些启动配置也在同一目录的 `sm70gui-config.json` 里，本来就只有
一份。页面自己的 localStorage 只是本机缓存，用来让首个画面不用等网络。

关窗口默认**最小化到托盘**（任务栏右下角）：程序、对话和 llama-server 都继续跑，
点托盘图标的“显示 / Show”恢复窗口，选“退出 / Quit”才真正结束进程并停掉它启动的
服务器。这个行为可以在**关于 → 设置**里关掉（“关闭窗口时最小化到托盘”，默认开启），
关掉后点 × 直接退出。同一个设置卡里还有“启动时自动加载已保存的模型”（默认关闭）：
打开后双击启动就直接把上次的模型加载起来；命令行 `--autostart` / `--no-autostart`
可临时覆盖这一次。

重复双击不会起第二个实例：检测到本机已有 launcher 时直接唤起它的窗口；后台模式没有
窗口，会提示到哪个地址访问。只想跑后台服务、不开窗口，用 `sm70gui.exe --no-browser`。
窗口版没有控制台，启动过程写在 `%TEMP%\sm70gui-launcher.log`，服务端日志在
`%TEMP%\sm70gui-server.log`，出问题先看这两个文件。

它不是一个新服务：一个本地进程同时把网页界面发到 `127.0.0.1`、代理
llama-server 的 API（流式响应逐块转发）、并管理 llama-server 的启停。所有数据只在
本机回环上。

### 第一次用

1. 先把 llama.cpp 这棵树编出来（`cmake --build build`），GUI 会自己找
   `<repo>\build\bin\llama-server.exe`；放到别处就用
   `sm70gui.exe --server-exe <路径>` 指一下。界面端口用 `--gui-port` 改，
   配置放哪用 `--config` 改（默认在 llama-server 旁边）；
2. 左起 **Launch** 页：`浏览…` 选模型目录（目录一变会自动重扫）再选模型文件；
   本地 `.gguf` 走 `-m`，填 `user/repo[:quant]` 则走 `-hf`；有视觉塔填
   `视觉投影文件（mmproj）`，独立草稿模型填 `投机解码文件`，`模型名称` 对应
   `--alias`；
3. **设备**：只列出 CUDA 驱动报告的卡，**只有计算能力 7.0（SM70）的卡能开**；
   `CUDA_VISIBLE_DEVICES` 由这些开关按 CUDA 序号拼出（与 nvidia-smi 顺序不同），
   非 SM70 卡不给开关；
4. **服务器参数**填常规参数（`-c/-np/-b/-ub/-t/-tb/-ngl/-sm/-fa/-ctk/-ctv/
   --port/--cache-ram/--spec-type/--spec-draft-n-max/--temp/--top-p/--top-k/
   --min-p/-n`），字段没覆盖的写进“额外参数”；
5. **环境变量**：常用 9 个直接列出，其余收在“高级”里，每条带默认值和双语说明；
   表里开的开关就是真正会导出的变量，下方命令行预览只显示与默认值不同的项；
6. 想重载或换参数点 **Start**（保存过模型的话，双击启动时已经自动加载）；
   启动过程与服务日志在同一页；`Stop` 只停本进程启动的服务器。

改动**即改即存**（停止输入 500 ms 后写盘），状态栏显示保存时间与文件路径；
`全部恢复默认` 会删掉配置文件回到内置默认值。**配置文件**卡片可以把当前表单存成
命名配置、读回、重命名、删除，或导出/导入成任意路径的 JSON。界面语言跟随系统，
可在 About 里强制 English / 简体中文；主题跟随 `data-theme`（浅色/深色）。

各页签：`对话` 支持流式输出、思考计时、首字延迟、文本附件与采样抽屉；`监控` 每秒
读一次 llama-server 的 `/metrics`、`/slots` 和本机 CPU/内存/磁盘；`关于` 显示运行中
模型、本机配置以及表单将要启动的模型/别名；`启动` 就是上面那张表单。

（维护者：改完 `tools/sm70gui/` 或 `tools/webui/` 后运行 `build_exe.bat`，它会
重新打包并刷新仓库里的 `tools/sm70gui/sm70gui.exe`；该文件随仓库提交，别人 clone
下来即可直接用。）

## 仓库结构

```
ggml/src/ggml-cuda/
├── fattn.cu                      # 选择钩子（cc==700 + D256 + 因果 + 预填充/解码宽度）
├── fattn-sm70-d256.cu            # 预填充 launcher：Q 暂存 / K,V 去量化 / SplitKV3 / scatter
├── fattn-sm70-d256-kernel.cuh    # Split-D N32 D256 kernel + SplitKV3 + q4-direct 分支
├── fattn-sm70-d256-decode.cuh    # D256 张量核心解码 kernel（GQA 打包的验证宽度）
├── allreduce-p2p.cu              # 单发 P2P allreduce（TP2 预填充路径）
├── dequant-sm70.cuh              # sm_70 iq3_s / iq3_xxs / iq4_xs 去量化 kernel
├── gated_delta_net.cu            # GDN kernel + opt-in l2norm prologue 融合

├── q4k-skinny.cu / iq-skinny.cu  # 瘦身权重布局（opt-in）
└── sm70-vendor/                  # cute/cutlass（Apache-2.0）+ flash（BSD-3）
verify/                           # harness、消融生成器、归因工具
bench/                            # 负载、扫描 CSV、画图脚本
docs/sm70-*.md                    # 解码移植计划、确定性归因、GDN 融合、
                                  # kernel 天花板、性能盘点
```

## 分支与同步

- `main`：上游 master + 插件（基线 2026-08-18 = ggml-org/llama.cpp
  `25ae3a9b3`；最后一次追平合并 2026-10-24，上游 `280141e6d4`）。
- 物理构建机只做 `git pull && cmake --build build`。
- 与上游的冲突仅限于 workflow 删除（本分叉不发布 docker/release 流水线）。

## 提交链（节选）

- `be7c77734` —— v1.0 钩子（管线验证，stock kernel）
- `00d7c8b57` —— v1.1 真正的 SM70 D256 Split-D kernel + vendor 闭包
- `284b251` —— **8/23 stride 修复**（0.69 根因；K/V 去量化 stride 是位置优先）
- `2722ed274` —— 注意力输出全链路 f32
- `b472e483f` + `fd718f946` —— **SplitKV3** 移植 + NaN 边界修复
- `c96bae56d` —— q4-direct 核内载入，裁决保持 opt-in
- `1921132d8` —— SplitKV3 merge 分母保护（10/13）
- `25caf6bec` / `a2e69d08e` / `0a6f6e8d0` —— D256 张量核心解码 kernel（WP1 P1/P2）
  与「只接管验证宽度」的路由，预取负结果归档
- `b32348e2e` —— Windows 工具链 + V100 TCC 切换
- `968e69a6d` —— 单发 P2P allreduce 移植
- `cb3d4806f` / `1a1148d0c` —— 自适应解码切分 / IQ 瘦身重排
- `40420489b` / `3b85a5dea` —— sm_70 去量化 kernel，收尾并在真机验证
- `76d71f7cc` / `da38b1363` —— step profiler（P0）+ GDN l2norm prologue 融合
  （P1，opt-in、逐位精确）与确定性定位
- `ac3c27910` —— **iq4_xs 去量化 LUT 越界修复**（确定性根因 1）
- `315dbb08e` —— **D256 预填充因果基准改取 mask**（根因 2）
- `627961dc0` / `597028032` / `4463259d1` / `17cc119fc` —— **P2P allreduce
  窗口、block 数、自暂存与 fence**（TP2 预填充 739 → 1558–1612）

## 致谢与许可

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) —— Split-D D256
  kernel 与 SplitKV3 patch 的上游来源
- [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100)
  —— FA2 基础层（BSD-3）
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) —— CuTe/CUTLASS 头文件
  闭包（Apache-2.0）
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) —— 宿主（MIT）
- WyvernTKC/llama.cpp-4xV100 —— 单发 P2P allreduce 设计来源（MIT），本移植
  做了本地化改造
- llama.cpp-v100 v100-1.1.1 —— 瘦身小 M GEMM 家族与 GDN 分块预填充 kernel 的
  来源（MIT，见 `ggml/src/ggml-cuda/LICENSE.v100-skinny`）

人机协作：AI 参与的提交带
`Co-authored-by: Hermes-Tianji <hermes-tianji@users.noreply.github.com>`
尾注。08-18..08-24 的提交（包括 9 行 stride 修复）与五个 2026-08-26 的提交早于
该约定，未带尾注；本行一并覆盖。

许可证：**MIT**，与上游 llama.cpp 一致（见根目录 `LICENSE`）。vendored 在
`ggml/src/ggml-cuda/` 下的第三方头文件保留各自许可，都是宽松许可、与 MIT 兼容：

| 路径 | 许可 |
|---|---|
| 根 `LICENSE` | MIT（ggml authors） |
| `sm70-vendor/cute`、`sm70-vendor/cutlass` | Apache-2.0（`sm70-vendor/LICENSE-cute-cutlass`） |
| `sm70-vendor/flash` | BSD-3（`sm70-vendor/LICENSE-flash-attention`） |
| `LICENSE.v100-skinny`（瘦身小 M GEMM、GDN 分块 kernel） | MIT |
