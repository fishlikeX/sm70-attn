#!/usr/bin/env python3
"""Timing-only ablation copies of the D256 prefill kernel.

Variants are written next to the production header as abl_<name>.cuh and
included via -DABL_HEADER_<NAME>; the production header is never modified.
Anchors are single-line prefixes (no embedded newlines) so CRLF/LF both work.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "ggml", "src", "ggml-cuda",
                   "fattn-sm70-d256-kernel.cuh")

I12 = " " * 12
I8 = " " * 8
I4 = " " * 4


def guard(indent, call):
    return (indent + call, indent + "if (false) " + call)


VARIANTS = {
    "nov": [guard(I12, "splitd_pv_gemm_tt(")],
    "noqk": [guard(I12, "FLASH_NAMESPACE::gemm<false, false>(")],
    "nogemm": [guard(I12, "FLASH_NAMESPACE::gemm<false, false>("),
               guard(I12, "splitd_pv_gemm_tt(")],
    "nomask": [guard(I8, "mask.template apply_mask<true, true>(")],
    "noexp": [guard(I4, "FLASH_NAMESPACE::sm70_scale_apply_exp2(")],
    "bm32": [("    static constexpr int kBlockM = 64;",
              "    static constexpr int kBlockM = 32;"),
             ("    static constexpr int kNThreads = 256;",
              "    static constexpr int kNThreads = 128;"),
             ("    static_assert(kSmemBytes == 45568);",
              "    static_assert(kSmemBytes == 26880);")],
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print("usage: ablate.py {%s}" % "|".join(VARIANTS))
        return 2
    name = sys.argv[1]
    with io.open(SRC, "r") as f:
        text = f.read()
    for old, new in VARIANTS[name]:
        n = text.count(old)
        if n != 1:
            print("anchor not unique/found for %s: %r (%d)" % (name, old[:60], n))
            return 1
        text = text.replace(old, new)
    out = os.path.join(HERE, "abl_%s.cuh" % name)
    with io.open(out, "w") as f:
        f.write(text)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
