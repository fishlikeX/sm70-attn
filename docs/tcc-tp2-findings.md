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

## 后续路线（按价值）

1. 移植 dual-v100 fork 的 matched-allreduce overlay（+3-4%）
2. 接受率调优（n_max/n_min 扫描、温度）；Q8_0 权重 × 高接受率组合复测
3. 修 meta 后端 handle_per_row 使 DFlash 图形可通过（或 fork 侧改 DFlash 图形避开
   axis-0 切分源）——DFlash 在 TP2 下的潜力未探明
4. WSL 注意：TCC 期间 V100 对 WSL2 不可见；切回 WDDM 用 tools/win/restore-wddm.cmd
