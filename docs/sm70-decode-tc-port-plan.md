# SM70 D256 Tensor-Core Decode Attention — 移植设计（WP1）

状态：**P1 已落地并通过正确性门禁（2026-10-24）**。P2（q4-direct）待做。

## P1 实测记录（V100-SXM2-32GB, WSL2, Q4_K_XL + q4_0 K + f16 V）

- 独立测试器 verify/sm70_dec_verify.cu（run_dec.sh）：9 case 全绿
  （w=1/2/8、splits 1..24、live<<容量 mask 语义、GQA 6/8；max|err| ≤ 1.1e-3）。
- logit A/B（贪心 128 tok）：零分叉（~6k 与 174,479 token 深上下文双档）。
- 性能（诚实记录，P1 为正确性/管线里程碑，非性能版）：
  ~6k ctx tg：stock 27.45 → tc 25.56（-7%）；174k ctx tg：10.07 → 9.39（-6.8%）。
  原因：P1 走 f16 K 镜像（174k 时每步 358MB 额外写+读），stock vec 直读 q4 K。
  P2 的 q4-direct 内核直读（复用 prefill 的 sm70_q4_fill_kv）即为消除此差额。
- 调试档案（5 个根因，逐个验证后修复）：PV f16 累加器布局（row=lane 连续 8 维，
  非 QK D-tile 映射）；partial split 平面跨度（跨全部 kv head 的全局行数）；
  merge rescale 漏乘 Log2E（exp2 域）；decode mask 满容量、活长度在 -inf 边界
  内容里（设备端扫描，顺带 CUDA-Graph 静态网格）；dst 维度序与 Q 相反
  （实测 (D,heads,q_len)），merge 步长按 ne 值内省推导。

## 1. 动机

- fork 现状：D256 tensor-core 内核只覆盖 prefill（`fattn-sm70-d256.cu` 拒绝
  `neq1 < 256` 的 ubatch）；decode 注意力在上游 `fattn-vec`（SIMT）上运行。
  README「不移植 XQA decode」的决策依据是 8k 上下文注意力占比 <0.1%。
- 该依据在深上下文失效：KV 读流量随上下文线性增长，65k-229k（生产容量）区间
  注意力重新变成 co-dominant；且 Windows TP2 落地后权重墙减半，注意力占比
  进一步放大。ninfer 实测（82k, width 4）：INT8 SIMT 1.86 vs bf16 TC
  2.71 TFLOP/s——TC 拓扑本身值 1.46×，且与 KV 压缩正交。

## 2. 研究来源（已读源码）

| 来源 | 文件 | 拿什么 |
|---|---|---|
| ninfer-V100X2 | `src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh`（723 行，头注释含完整推导） | **整体拓扑**：mma.sync.m8n8k4、DimSplit=WarpsPerCta（4/8）切 head dim、Br=32/Bc=16、Q 驻留 smem（独立 q_s/k_s/v_s，32KB 静态）、寄存器 staging 反量化喂 mma、partial_acc/m/l 输出接现成 merge |
| xormal/V100-FA2（HOOLIGAN, MIT） | `fa2_sm70/csrc/split_decode.cu`（split_defer_mqa_kernel） | **SIMT 侧互补机制**：GF query-head 组共享单 KV head 读（GQA 6:1 → 6× 读复用）、设备端按 seqlens 定 split 界（CUDA Graph 安全静态网格）、FA2SM70_DEC_VK V/K 同拍发射（在途 load ×2，实测 +16%/+28% @32k/256k）、投机 verify 尾掩码 + qGrp 虚拟批（免拷贝） |
| 1CatAI/1Cat-vLLM | `csrc/attention/sm70_grouped_long` | packed 多 query/head 共享 KV 读的 grouped 组织、FP32 分区状态精度边界（其设计文档区分存储量化/反量化算术/归约三重误差） |
| kamilch1k/voltattn | `src/voltattn.cu` | 8-lane/位置、4 位置/warp 的访存并行组织（D128 几何，机制参考） |

负面结论同样约束设计（1Cat TP2 verifier 记录 + ninfer 调研转述）：
多个 unroll/寄存器限制/预取变体实测**变慢**——所有机制都按候选对待，逐项 A/B。

## 3. 设计

### 接入点
- 新文件 `ggml/src/ggml-cuda/fattn-sm70-d256-decode.cu(h)`。
- 路由：`fattn.cu` 的 dispatch 在 `cc == 70 && D == 256 && causal && neq1 < 256`
  时走新内核；env `LLAMA_SM70_D256_DECODE`（默认 0，验收后转 1），
  与 prefill 内核的 `LLAMA_SM70_D256` 独立。
- KV dtype：首发 f16 V + q4_0 K（生产组合），复用 prefill 内核的
  `sm70_q4_fill_kv` 反量化（q4-direct 轴后续再开）。

### P1（ninfer 拓扑直译）
- 4 warp DimSplit（每 warp 64 维 PV 输出），Br=32 行 tile 覆盖
  (TokenTile × GroupSize)（GQA 6:1，nq≤5 时单 pass；宽 verify 两 pass），
  Bc=16 key/tile，q_s(16KB)+k_s(8KB)+v_s(8KB) 静态 smem。
- QK：两 warp 算两个 8-key QK tile 共享 FP32 分数（ninfer 的共享结构）；
  PV：volta mma.m8n8k4，exp2 online-softmax。
- 输出 partial 格式与既有 SplitKV3 merge kernel 对齐（partial_out/max/sum），
  merge 复用 `fattn-sm70-d256.cu` 已有实现。

### P2（HOOLIGAN 机制叠加）
- GQA 打包：CTA = 1 KV head × GF query heads，K/V 每字节读一次供 6 行用。
- V/K 同拍发射（DEC_VK 思路）。
- split 界由 seqlens 设备端推导（静态网格，Graph 安全）。

### P3（投机协同）
- verify 场景（neq1 = 草稿数+1）用尾掩码 + 虚拟批，免表拷贝（qGrp）。

## 4. 验收门（沿 fork 既有纪律）
1. `verify/sm70_verify.cu` 新 case：decode 形状（nq=1/2/8、kv=3000/32768、
   q4K 轴、GQA 6:1），CPU 参照同门槛（>1e-2 计数、nan/inf、top-err 报告）。
2. run.sh 钉值扩展：decode 内核文件单独一行 md5 门。
3. logit A/B：贪心 128 token，ON vs stock 零分叉（同 `verify/logit_q4direct.sh`）。
4. 性能 A/B：8k / 85k / 176k 三档 tg（同 `perf_ab_176k.sh` 方法论），
   发布阈值：深档 ≥ +10% 才转默认 on。

## 5. 风险
- 寄存器墙：ninfer 推导（255/thread 上限）直接套用；若 P2 的 GF=6 打包
  超墙，回退 GF=3×两 pass。
- smem：48KB 静态上限内已论证；动态 smem 备选（cudaFuncSetAttribute）。
- fattn-vec 的 DKB 路径含按 dtype 的 K 反量化分派，新内核需对齐其
  mask/sink 处理语义（attention_sink 存在与否）。
