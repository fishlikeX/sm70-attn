# TCC + TP2 实测与全栈解锁（2026-10-25）

Windows 11 / 2×V100-SXM2-32GB（TCC + NVLink 6 链路）/ RTX 3080 (WDDM，不动)。

## NVLink 互联（TCC 下 P2P 探针，D:\zcode-build\p2p_bw.cu）

- P2P 双向可用；单向 145.1/144.9 GB/s、双向聚合 289.6 GB/s（NVLink 2.0×6 理论 300，98%）
- 小消息（4-64KiB cudaMemcpyAsync）~31 µs/条 —— 驱动提交开销，非链路延迟
- 坑：TCC 后 CUDA 枚举序变为 [V100, V100, 3080]，nvidia-smi 的 1,2 不再对应两张 V100；
  Windows 侧须 CUDA_VISIBLE_DEVICES=0,1（或干脆 0,1,2 内自选）。CUDA_DEVICE_ORDER=
  PCI_BUS_BUS_ID 在该驱动/TCC 组合下使 cudaGetDeviceCount 报 invalid ordinal，勿用。

## TP2 矩阵（-sm tensor -fit off，Q4_K_XL，174,479-token prompt）

| 配置 | decode tok/s | 备注 |
|---|---:|---|
| 单卡 f16V | 15.1 | 基线 |
| TP2 WDDM（浅档，f16V） | 30.3 | 旧测 |
| **TP2 TCC（浅档，f16V）** | **38.9** | vs WDDM +28%、vs 单卡 +35% |
| TP2 TCC（深档，f16V） | 18.5 | |
| TP2 TCC（深档，q4/q4） | 22.5 | |
| 单卡 DFlash2 spec q4/q4 | 24.0 | 上一代生产冠军 |
| **TP2 TCC × draft-mtp(3) × q4/q4 × TC verify 内核** | **33.18** | **生产全栈**；accept 0.55/mean 2.64 |

prefill：TP2 492-509 tok/s（单卡 ~470）。

## 关键发现：meta 后端 × 投机解码

- **draft-dflash × -sm tensor：崩**。`ggml-backend-meta.cpp:547` 的 handle_per_row
  断言（DFlash 图形某源被 axis-0 切分喂给按行算子），模型加载期退出。上游限制，
  非本 fork 引入。
- **draft-mtp × -sm tensor：完全可用**（本页数字即证）。MTP 头内置在 GGUF
  （blk.64.nextn.*），`--spec-type draft-mtp --spec-draft-n-max 3` 无需草稿文件。
- TC verify 内核在 meta 分片下正确接管（DEBUG probe：w=4 分片态 Q=(256,4,12,1)/K=2头
  与全视图 Q=(256,4,24,1) 均 ACCEPT；q_len=9 的草稿采样批正确回退 stock）。
- 参照系：123123213weqw/dual-v100-llama.cpp（thin-patch fork）TP2×MTP3×Q8_0×F16KV
  @116k = 43.34 tok/s、DFlash2 候选 33.0（MTP 为其生产默认）+ 自研 matched-precision
  allreduce overlay（+3-4% @32K-255K）。与我们的 33.18@174k 差距可由上下文长度
  （KV 扫描 116k vs 174k）、权重（Q8 接受率通常更高）、通信 overlay 解释——均为
  可追赶项，非原理差距。

## 追加（同日晚）：GSQ 生产权重 + n_max 扫描 + 层切分判死

- **生产权重纠正**：GSQ-RCO-IQ3_S-mtp（11.2GB）为 3.8 时代最优 27B 权重（维护者定论），
  Q4_K_XL 此前仅为基准惯性。全栈三点：
  单卡 GSQ 26.72（vs Q4 单卡 24.04，+11%）；层切分 GSQ 26.73（**判死**：decode 零增益，
  同步层链无重叠；prefill 倒是 +39%）；**TP2×GSQ = 34.99 tok/s @174k（当前生产冠军）**，
  接受率 0.61/mean 2.79（好权重连接受率都更高）。
- n_max 扫描：4=31.75 < 3=33.18（Q4 下），3 为最优；接受率 0.55-0.61 撑不起 4。
- meta×GSQ 断言（bufs[i] 空）：满上下文复测未复现（fprintf 级重编后两次加载皆成功），
  判为偶发/竞态观察项；若再现，按 meta_buf_ctx->bufs[i] 分配路径查 IQ3_S 切片。
- 层切分负结果的原因记录：decode 的小批量同步层链无流水重叠；勿再投入。

## 生产建议（当前）

深上下文生产（34.99 tok/s 配方）：Windows TCC + `D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
+ `-sm tensor -fit off -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3`
+ `LLAMA_SM70_D256_DECODE=1 LLAMA_SM70_D256_DEC_Q4=1`（模型本地盘；CUDA_VISIBLE_DEVICES=0,1）。

## 追加（同日夜）：ubatch 调优 + WyvernTKC 单发 P2P allreduce 移植

- **-ub 调优（零代码）**：根因=每 ubatch 全量权重 dequant（nsys 画像 ~1960 实例实证）。
  中档 30k：ub8192 prefill 662（+19%）decode 78.85；深档 ub512 仍最优（34.99 基线）。
- **P2P allreduce 移植**（WyvernTKC/llama.cpp-4xV100，MIT，allreduce-p2p.cu/.cuh）：
  单发对端写 LL 内核，接在 GGML_CUDA_ALLREDUCE 选择器之后、internal 链之前，
  仅认领小连续 F32 消息（其余回退）。本地化改动：fold_args 结构注入 .cuh、MoE sigmoid
  就地替身。**实测：30k decode 78.85 → 90.60（+14.9%）；174k decode 34.99 → 37.37（+6.8%）**。
- 最优配方分场景：深档 ub512+p2p=37.37；中档 ub8192+p2p=90.60；prefill 大 ub 662-678。
- 今日 174k 全阶梯：15.13 → 24.04 → 34.99 → **37.37**（+147% vs 晨间起点）。

## 追加（同夜深）：GDN concat 融合尝试 — 受阻归档

- concat_non_cont 的真实来源**不是** wqkv bias（GSQ 的 GGUF 本就带融合 attn_qkv_b，
  图构建器的 concat 分支不执行），而是 **MTP 头的 eh_proj**（qwen35/qwen35moe/qwen3next
  的 `ggml_concat(e_norm, h_norm)` + eh_proj matmul，每 MTP 轮一次）。
- 三次尝试（view 步长修正 → cont 输入归一化）均卡在 `ggml_can_repeat` 断言：
  build_hadamard_input 对连续/跨步输入走不同分支，QUASAR 系权重的 Hadamard 折叠
  使两个子 GEMV 的输出形状不对称，需要逐 artifact 适配（rot 表按权重走）。
  收益上限 ~1.3%/轮，按投入产出纪律**挂起**；loader 融合机制（bias_fuse 系列）
  因 GGUF 查名机制限制同样搁置，留档可复活。
- 若复活：从 rot 表对 eh_proj 的具体 perm 入手，或上游化一个 concat-elimination
  pass 到 ggml 图优化器。

## 追加（深夜二）：decode 深度曲线定罪 + 自适应 splits

- **深度三点扫描**（spec+q4/q4+p2p，24 splits）：30k=90.6 → 100k=56.6 → 174k=37.4。
  边际 KV 扫描带宽 ~150GB/s = **NVLink+HBM 池的 9%**，线性、非延迟地板。
- **根因：CTA 饥饿**。splits=24×4 头 = 96 CTA（每 SM 一个 4-warp CTA，25% 占用），
  每个 CTA 串行走 7k+ key-tile。env `LLAMA_SM70_D256_SPLITS` 开放 CTA 预算扫描：
  100k：24=56.75 → 48=62.17（+9.5%）→ 96=61.14（平台）；174k：24=37.37 →
  **48=43.47（+16.3%，当前深档冠军）** → 96=43.26（平台）。
- **自适应公式落码**：splits = clamp(kv_len/920, 1, cap)，cap 默认 48（env 可到 256）。
  深档验证 43.27（=手动 48 带内）；中档两轮 80.7/79.7 vs 基线 90.6——**中档回归未解**，
  见下。
- **ub 反向之谜（机制未明，数据双倍验证）**：中档 ub8192 = 86.1 vs ub512 = 79.65
  （+8%）；深档 ub8192 = 34.25 vs ub512 = 43.47（-21%）。ub 对 decode 的作用随深度
  反向，与 CUDA graph 捕获形态或 KV 缓冲布局相关，待 profile。
- **分场景最优配方**：深档（≥50k）splits48+ub512 = **43.47**；中档 ub8192 = 90.60
  （splits 用默认或 env 24）；prefill ub8192 = 662-678。
- 今日 174k 阶梯：15.13 → 24.04 → 26.72 → 33.18 → 34.99 → 37.37 → **43.47（+187%）**。
  116k 等效 ≈ 48-52（dual-v100 参照 43.34 @116k，已反超）。

## 追加（同夜三）：IQ skinny 重打包解锁 — 47.27 tok/s @174k

- **发现**：fork 自带完整的 IQ skinny 基础设施（iq-skinny.cu/.cuh，v100-skinny T2 线）：
  侧缓冲重打包（IQ3_S +7.3% 字节 ≈ +0.8GB）、f16 网格 LUT（1.5 ops/value 解码）、
  18.9M 值 1 ulp 验证、mul_mat 路由已接（is_repacked 优先）。**默认关**，且 IQ 类型
  需要双 env：`LLAMA_SM70_SKINNY=2` + `LLAMA_SM70_IQ_SKINNY=1`（缺一不触发）。
- **实测**（GSQ，TP2，splits48，ub512，174k）：prefill 437.12 → **467.94（+7%）**；
  decode 43.47 → **47.27（+8.7%）**；接受率 0.70/mean 3.06（轨迹方差内，无劣化）。
  同时加速两条路的原因：重打包记录的 LUT 解码（1.5 ops/value）比原 IQ3_S 块的
  查表+分支+乘法便宜，prefill 的 dequant 内核与 decode 的 GEMV 都吃到。
- 仅 skinny=2 无 IQ=1：无效果（IQ 类型路由门未开）——双 env 缺一不可。
- **生产配方更新**：追加两个 env。精度属性：1 ulp（非逐位一致），与 TC decode
  同一验收标准（harness + 接受率 + 输出质量）。
- 今日 174k 阶梯：15.13 → 24.04 → 26.72 → 33.18 → 34.99 → 37.37 → 43.47 →
  **47.27（+212%）**。prefill：470 → **468-467.94**（持平——prefill 与 decode
  首次同时受益）。

## 追加（同夜四）：prefill 叠满终点 + ub16384 判死

- **深档叠满**（ub8192+skinny2+IQ+splits48）：prefill **501.67**（基线 437 = +14.8%）。
  中档叠满 674（≈基线——22k 下 dequant 已被 ub8192 摊薄，skinny 无增量）。
- **ub16384 判死**：meta 缓冲分配 OOM（meta.cpp:1744 bufs[i] 空）——激活工作区翻倍
  挤爆 32GB 卡。ub8192 为该权重的 prefill 上限。
- **prefill 公平对标**（按 FLOP 计效率）：174k 全档工作量 = 线性 9.4 + attention 二次项
  ~6 = 15.4 PFLOP / 348s = **44.3 TFLOPS 有效**；dual-v100 833@117k = 9 PFLOM/140s =
  64 TFLOPS。公平差距 1.44×，来源 = IQ3_S 解包 ALU（skinny mode 2 的核内展开只省了
  搬运未省 ALU）+ cutlass 形状效率。
- **下一档的唯一大石头**：IQ3_S dequant 融合进 TC GEMM 内核（Marlin 式，vLLM 提取线
  格式不通需自写；dequant 数学须逐位对齐 convert.cu——q4 布局审计的教训适用）。
  这是多日内核工程，等待决策。

## 后续路线（按价值）

1. 移植 dual-v100 fork 的 matched-allreduce overlay（+3-4%）
2. 接受率调优（n_max/n_min 扫描、温度）；Q8_0 权重 × 高接受率组合复测
3. 修 meta 后端 handle_per_row 使 DFlash 图形可通过（或 fork 侧改 DFlash 图形避开
   axis-0 切分源）——DFlash 在 TP2 下的潜力未探明
4. WSL 注意：TCC 期间 V100 对 WSL2 不可见；切回 WDDM 用 tools/win/restore-wddm.cmd
