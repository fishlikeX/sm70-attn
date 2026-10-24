# q4_0 布局审计：prefill q4-direct 的 nibble 映射与 ggml 规范相悖（工作树草稿，未提交）

日期：2026-10-24。结论先行：**fork 的 prefill q4-direct（`sm70_q4_dequant_one/group`）与
harness CPU 参照（`deq_q4_one`）按"顺序映射"解码（byte b 低半字节=元素 2b、高=元素 2b+1），
而 ggml 全链路是"交错映射"（byte b 低=元素 b、高=元素 b+16）。对真实 q4_0 KV cache，
该路径会把 16..31 号维度读进 1..15 号位置（偶数维幸存、奇数维错位）。默认关，生产未受影响；
但 opt-in 开启时输出错误。同时 harness 的 fixture 写入器是规范布局，与自己的参照/内核不一致
——q4 系列 case 的"23/23"是内核与错误参照的互证，对布局盲。**

## 证据链（三处独立源码，互相一致）

1. **规范写入（参考实现）** `ggml/src/ggml-quants.c` `quantize_row_q4_0_ref`：
   `x0 = x[j]`（低半字节）、`x1 = x[qk/2 + j]`（高半字节），j ∈ [0,16)。
2. **GPU 缓存实际写入端** `ggml/src/ggml-cuda/cpy-utils.cuh`（set-rows.cu:261 对
   `dst->type == GGML_TYPE_Q4_0` 调 `quantize_f32_q4_0_block`）：
   `x0 = x[0 + j]`、`x1 = x[QK4_0/2 + j]` —— 同一交错布局。KV cache 的量化写即经此路径。
3. **规范读取（镜像）** `ggml/src/ggml-cuda/convert.cu` `dequantize_block_q4_0`：
   `y[l+0] = q[l] & 0xF`（元素 l）、`y[l+16] = q[l] >> 4`（元素 l+16）—— 一致。

## fork 侧的顺序映射（与上述相悖）

- `fattn-sm70-d256-kernel.cuh` `sm70_q4_dequant_one`：`pair = b[2 + ((d&31)>>1)]`，
  `(d&1) ? pair>>4 : pair&0xF` —— 元素 d 取字节 d/2 的奇偶半字节（顺序）。
- 同文件 `sm70_q4_dequant_group`：`out[j] = ((nib >> 4j) & 0xF)`，nib 为连续字节流 —— 同顺序。
- `verify/sm70_verify.cu` `deq_q4_one`（CPU 参照）：与内核同式 —— 参照同错。
- 同文件 `quant_row_q4_0`（fixture 写入器）**却是规范布局**（`x[j]`/`x[16+j]`）—— 写读不一致，
  这就是 q4 case 全绿却测不出布局错误的原因。
- 落地提交 `c96bae56de`（q4-direct）晚于最后一次静态审计态 `daca614`（md5 钉值锚点），
  即该映射从未经过"钉值态"复核；`logit_q4direct.sh` 的 D-vs-S 逐位一致门在落地后是否真实
  通过，本审计无法从仓库复核（需真机重跑）。

## 修复（本工作树已改，待真机复验）

- decode 侧（本分支新增）：`fattn-sm70-d256-decode.cuh` 的 `sm70_dec_q4_dequant8`
  已按规范交错布局实现（8 对齐 chunk 不跨 16 界：d&16==0 全低半字节，否则全高半字节；
  4 次对齐 u16 装载拼 8 字节）。实测：顺序版在真服务器上贪心输出立刻分歧（近并列 logit
  翻转），规范版待卡复验零分叉。
- prefill 侧：`sm70_q4_dequant_one` / `sm70_q4_dequant_group` 改为规范映射（本次一并修）。
- harness 侧：`deq_q4_one` 改规范；fixture 写入器本就规范，不动。q4 系 case 需真机重跑。

## 待办（需 GPU）

1. `bash verify/run.sh`（q4 系 case 以修正后的参照重跑，应全绿）。
2. `bash verify/logit_q4direct.sh`：D-vs-S 逐位一致门，现在才是有效门。
3. decode：P2 真机 A/B 复验（修正版）。
