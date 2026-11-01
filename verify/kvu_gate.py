"""End-to-end gate for the mask-driven SM70 prefill path under a unified KV pool.

Starts llama-server with -kvu -np N, drives a fixed set of greedy requests in a
fixed pattern, and compares two runs:
  A: LLAMA_SM70_D256=1  (mask-driven prefill, decode on stock)
  B: LLAMA_SM70_D256=0  (stock everywhere)

Two comparisons:
  gate 1b  A(burst) vs B(burst)   - same request pattern, different kernel.
  gate 2   burst vs solo, on each build separately, with the stock run as the
           baseline for how much drift concurrent batching causes by itself.

What can actually be asserted, measured on this box:

  * greedy ids are a HARD assertion only when the composition is reproducible,
    i.e. --prompts same (identical prompts, all slots in lockstep). With
    different lengths the server batches differently from run to run and the
    greedy stream can move - for the stock build too (measured: stock moved 2
    of 4 bursts by 13 tokens once the timing shifted). So --prompts hetero
    reports the movement of both builds instead of gating on it, and the
    numbers are the point.
  * logprobs are judged against the drift the same build shows against itself,
    not against a fixed constant: a fixed 1e-3 fails stock against itself.
  * a rank swap only counts when the swapped pair sits further apart than that
    drift.

Usage:
  python verify/kvu_gate.py --model <gguf> --mode burst --pattern same
  python verify/kvu_gate.py --model <gguf> --mode burst --pattern hetero --tp 2

Exit code 0 = pass.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "build", "bin", "llama-server.exe")

FILLER = (
    "The unified key-value pool lets several conversations share one buffer. "
    "Each sequence owns a scattered set of cells, and the attention mask is the "
    "only record of which cells a query may read. "
)


def make_prompt(n_paras, tag):
    body = "".join(f"{tag}-{i}: {FILLER}" for i in range(n_paras))
    return body + f"\nReply with the single word OK and then stop.\n"


class Server:
    def __init__(self, args, env_extra, tag):
        self.port = args.port
        self.log_path = os.path.join(args.logdir, f"server_{tag}.log")
        env = dict(os.environ)
        env.update(env_extra)
        argv = [
            BIN, "-m", args.model, "-ngl", "99", "-fit", "off",
            "-c", str(args.ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
            "-fa", "on", "-ctk", args.ctk, "-ctv", args.ctv,
            "-t", "12", "-tb", "16", "--temp", "0",
            "-np", str(args.np), "-kvu", "--seed", "42",
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        if args.tp == 2:
            argv += ["-sm", "tensor"]
        if args.mtp:
            argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.mtp)]
        if args.decode:
            env["LLAMA_SM70_D256_DECODE"] = "1"
            env["LLAMA_SM70_D256_DEC_Q4"] = "1"
        if args.debug:
            env["LLAMA_SM70_D256_DEBUG"] = "1"
        if args.cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        for kv in args.env:
            k, _, v = kv.partition("=")
            env[k] = v
        self.argv = argv
        self.env = env
        self.proc = None

    def start(self, deadline=600):
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log = log
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early, see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("server never became healthy")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self._log.close()
            self.proc = None

    def ledger(self):
        """Aggregate the routing probe lines by reason."""
        out = {}
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError:
            return out
        for m in re.finditer(r"\[sm70-d256\] \S+ (\S[^|]*?) \|", body):
            reason = m.group(1).strip()
            out[reason] = out.get(reason, 0) + 1
        return out


def complete(port, payload, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_pattern(port, prompts, pattern, n_predict, n_probs):
    """Issue the prompts in the requested pattern, return one result per prompt."""
    def payload(p):
        return {"prompt": p, "n_predict": n_predict, "temperature": 0.0,
                "top_k": 1, "cache_prompt": False, "n_probs": n_probs}

    results = [None] * len(prompts)
    if pattern == "solo":
        for i, p in enumerate(prompts):
            results[i] = complete(port, payload(p))
        return results

    if pattern == "burst":
        barrier = threading.Barrier(len(prompts))

        def worker(i):
            barrier.wait()
            results[i] = complete(port, payload(prompts[i]))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    if pattern == "stagger":
        def worker(i):
            time.sleep(1.5 * i)
            results[i] = complete(port, payload(prompts[i]))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    raise ValueError(pattern)


def ids_and_probs(resp):
    ids, lps = [], []
    for e in resp.get("completion_probabilities") or []:
        ids.append(e["id"])
        top = sorted(e["top_logprobs"], key=lambda t: -t["logprob"])
        lps.append([(t["id"], t["logprob"]) for t in top])
    return ids, lps


def compare(a, b, tol, flip_thr=None):
    """Return (token_mismatches, max_abs_dlogp, rank_flips, ties)."""
    if flip_thr is None:
        flip_thr = tol
    ia, la = ids_and_probs(a)
    ib, lb = ids_and_probs(b)
    mism = sum(1 for x, y in zip(ia, ib) if x != y) + abs(len(ia) - len(ib))
    max_d = 0.0
    flips = 0
    ties = 0
    for ta, tb in zip(la, lb):
        da = dict(ta)
        db = dict(tb)
        for k in set(da) & set(db):
            d = abs(da[k] - db[k])
            if d > max_d:
                max_d = d
        ra = [k for k, _ in ta]
        rb = [k for k, _ in tb]
        if ra != rb:
            # A rank change only counts when the swapped pair sits further apart
            # than the run-to-run noise; inside that band the order is not
            # meaningful. flip_thr carries that band (tol for exact comparisons).
            swapped = [(k, da[k] - db[k]) for k in set(da) & set(db) if ra.index(k) != rb.index(k)]
            if any(abs(delta) > flip_thr for _, delta in swapped):
                flips += 1
            else:
                ties += 1
    return mism, max_d, flips, ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--ctk", default="q4_0")
    ap.add_argument("--ctv", default="q4_0")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--cuda-devices", default="")
    ap.add_argument("--pattern", default="burst", choices=["burst", "stagger"])
    ap.add_argument("--prompts", default="same", choices=["same", "hetero", "long", "gen", "genlong"],
                    help="same: identical prompts, so every slot sees the same "
                         "composition and the greedy stream is a hard assertion. "
                         "hetero: different lengths, the composition legitimately "
                         "differs between runs and the tokens are a measurement")
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--decode", action="store_true", help="enable the D256 decode kernel env")
    ap.add_argument("--mtp", type=int, default=0,
                    help="speculative draft-mtp with this n-max (0 = off)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE passed to the server process, repeatable "
                         "(e.g. --env LLAMA_SM70_SKINNY=2)")
    ap.add_argument("--logdir", default=os.environ.get("TEMP", "/tmp"))
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--repeat", type=int, default=0,
                    help="also run the pattern N more times per build and report how often "
                         "the greedy stream moves against the first run")
    args = ap.parse_args()

    if args.prompts == "same":
        texts = [make_prompt(40, "S")] * args.np
    elif args.prompts == "gen":
        # Generation-heavy: the decode kernel is only measurable when the model
        # actually emits more than the stop token. No "reply with OK" tail here,
        # or the model says OK and stops.
        filler = "".join(f"G-{i}: {FILLER}" for i in range(20))
        texts = [filler + "\nList the numbers from 1 to 120, one per line, then stop.\n"
                 for _ in range(args.np)]
    elif args.prompts == "genlong":
        # long context + real generation: the decode kernel's home turf
        filler = "".join(f"X-{i}: {FILLER}" for i in range(380))
        texts = [filler + "\nList the numbers from 1 to 120, one per line, then stop.\n"
                 for _ in range(args.np)]
    elif args.prompts == "long":
        # > n_batch, so one slot's prefill spans several ubatches and the slots'
        # chunks interleave in them - the most fragmented cell layout there is.
        texts = [make_prompt(230 + 6 * i, "L") for i in range(args.np)]
    else:
        texts = [make_prompt(40 + 15 * i, "H") for i in range(args.np)]

    runs = {}
    burst_wall = {}
    for tag, env_extra in (("tuned", {"LLAMA_SM70_D256": "1"}),
                           ("stock", {"LLAMA_SM70_D256": "0"})):
        srv = Server(args, env_extra, tag)
        srv.start()
        try:
            # Warm-up: the weights are mmap'd, so whichever build runs first
            # pays the page-in of ~11 GB inside its first prefill. Without this
            # the first build looks 10x slower than it is (measured: 28.4 s vs
            # 2.7 s for the same 1682-token prompt).
            complete(srv.port, {"prompt": "warm up", "n_predict": 1, "temperature": 0.0,
                                "cache_prompt": False})
            complete(srv.port, {"prompt": make_prompt(6, "W"), "n_predict": 1,
                                "temperature": 0.0, "cache_prompt": False})
            solo = run_pattern(srv.port, texts, "solo", args.n_predict, args.n_probs)
            solo2 = run_pattern(srv.port, texts, "solo", args.n_predict, args.n_probs)
            t0 = time.time()
            burst = run_pattern(srv.port, texts, args.pattern, args.n_predict, args.n_probs)
            burst_wall[tag] = time.time() - t0
            burst2 = run_pattern(srv.port, texts, args.pattern, args.n_predict, args.n_probs)
            extra = [run_pattern(srv.port, texts, args.pattern, args.n_predict, args.n_probs)
                     for _ in range(args.repeat)]
            runs[tag] = {"solo": solo, "solo2": solo2, "burst": burst, "burst2": burst2,
                         "extra": extra, "ledger": srv.ledger(),
                         "tokens": [r["tokens"] for r in burst],
                         "tps": [r["timings"]["prompt_per_second"] for r in burst]}
        finally:
            srv.stop()

    print("\n=== routing ledger ===")
    for tag in runs:
        print(f"[{tag}]")
        for k, v in sorted(runs[tag]["ledger"].items(), key=lambda kv: -kv[1]):
            print(f"   {v:6d}  {k}")

    # Control: how reproducible is a run at all? Without this number no
    # cross-run comparison means anything.
    print("\n=== control: same build, same pattern, run twice ===")
    for tag in runs:
        for pat in ("solo", "burst"):
            wm = wd = wf = 0
            for i in range(len(texts)):
                m, d, f, _ = compare(runs[tag][pat][i], runs[tag][pat + "2"][i], args.tol)
                wm, wd, wf = max(wm, m), max(wd, d), max(wf, f)
            print(f"  [{tag}/{pat}] tok_mismatch={wm} max|dlogp|={wd:.2e} rank_flips={wf}")

    # The server-level patterns are not reproducible run to run (the control
    # above measures it), so a fixed 1e-3 cannot be the bar here: the bar is the
    # drift the stock build itself shows under the same pattern. Token identity
    # is still absolute, at every level.
    noise = 0.0
    for tag in runs:
        for i in range(len(texts)):
            _, d, _, _ = compare(runs[tag]["burst"][i], runs[tag]["burst2"][i], args.tol)
            noise = max(noise, d)
    thr = max(args.tol, 2.0 * noise)
    # Rank swaps the same build shows against itself, at the same significance
    # threshold the tuned-vs-stock comparison uses. A decode-heavy workload can
    # move the top-5 by several nats between runs, and then a handful of swaps is
    # the expected count, not a defect - the control is what says how many.
    ctrl_flips = 0
    for tag in runs:
        for i in range(len(texts)):
            _, _, f, _ = compare(runs[tag]["burst"][i], runs[tag]["burst2"][i],
                                 args.tol, flip_thr=noise)
            ctrl_flips = max(ctrl_flips, f)
    print(f"\n  scheduling noise floor (same build, same pattern, twice): {noise:.2e} "
          f"-> threshold {thr:.2e}, control rank swaps at that threshold: {ctrl_flips}")

    if args.repeat:
        print(f"\n=== repeat: {args.repeat + 2} bursts per build, each against the first ===")
        for tag in runs:
            seqs = [runs[tag]["burst"], runs[tag]["burst2"]] + runs[tag]["extra"]
            moved = 0
            for k in range(1, len(seqs)):
                worst = max(compare(seqs[k][i], seqs[0][i], args.tol)[0] for i in range(len(texts)))
                if worst:
                    moved += 1
                print(f"  [{tag}] run {k}: worst tok_mismatch vs run 0 = {worst}")
            print(f"  [{tag}] bursts that moved: {moved}/{len(seqs) - 1}")

    ok = True
    md_accept = sum(v for k, v in runs["tuned"]["ledger"].items() if "mask-driven" in k)
    if md_accept == 0:
        ok = False
        print("  <-- FAIL: the tuned run never selected the mask-driven kernel, this test proves nothing")
    else:
        print(f"  mask-driven prefill nodes in the tuned run: {md_accept}")

    hard_tokens = (args.prompts == "same")
    if not hard_tokens:
        print("\n  (hetero prompts: greedy movement is reported for both builds, not gated)")

    print("\n=== gate 1b: tuned vs stock, same pattern (%s) ===" % args.pattern)
    for i in range(len(texts)):
        mism, max_d, flips, ties = compare(runs["tuned"]["burst"][i], runs["stock"]["burst"][i],
                                           args.tol, flip_thr=noise)
        bad = (hard_tokens and mism > 0) or max_d > thr or flips > max(ctrl_flips, 1)
        ok = ok and not bad
        print(f"  req {i}: prompt_n={runs['tuned']['burst'][i]['timings']['prompt_n']:5d} "
              f"tok_mismatch={mism:3d} max|dlogp|={max_d:.2e} rank_flips={flips} ties={ties}"
              f"{'   <-- FAIL' if bad else ''}")

    if not args.skip_solo:
        print("\n=== gate 2: concurrent vs solo (stock drift is the baseline) ===")
        for tag in ("stock", "tuned"):
            worst_m = worst_d = worst_f = 0
            for i in range(len(texts)):
                m, d, f, _ = compare(runs[tag]["burst"][i], runs[tag]["solo"][i], args.tol)
                worst_m = max(worst_m, m)
                worst_d = max(worst_d, d)
                worst_f = max(worst_f, f)
            print(f"  [{tag}] worst: tok_mismatch={worst_m} max|dlogp|={worst_d:.2e} rank_flips={worst_f}")
        base_m, base_d, base_f = 0, 0.0, 0
        for i in range(len(texts)):
            m, d, f, _ = compare(runs["stock"]["burst"][i], runs["stock"]["solo"][i], args.tol)
            base_m, base_d, base_f = max(base_m, m), max(base_d, d), max(base_f, f)
        for i in range(len(texts)):
            m, d, f, _ = compare(runs["tuned"]["burst"][i], runs["tuned"]["solo"][i],
                                 args.tol, flip_thr=base_d)
            tok_bad = hard_tokens and m > base_m
            if tok_bad or d > max(base_d * 3.0, args.tol) or f > max(base_f, 1):
                ok = False
                print(f"  req {i}: tuned drift ({m},{d:.2e},{f}) exceeds stock baseline "
                      f"({base_m},{base_d:.2e},{base_f})   <-- FAIL")

    print("\n=== throughput (burst wall clock from the client side) ===")
    n_prompt = sum(r["timings"]["prompt_n"] for r in runs["tuned"]["burst"])
    for tag in ("stock", "tuned"):
        wall = burst_wall[tag]
        dec = [r["timings"]["predicted_per_second"] for r in runs[tag]["burst"]]
        print(f"  [{tag}] {n_prompt} prompt tokens + {args.n_predict} generated x{len(texts)} "
              f"in {wall:.1f}s -> {n_prompt / wall:7.0f} prompt tok/s aggregate "
              f"(per request: " + " ".join(f"{t:.0f}" for t in runs[tag]["tps"]) + ")")
        print(f"  [{tag}] decode per request: " + " ".join(f"{d:.1f}" for d in dec) + " tok/s")
        # Solo decode = one request in flight, so no slot contention: the only
        # number that compares the two kernels rather than the scheduler.
        dsolo = [r["timings"]["predicted_per_second"] for r in runs[tag]["solo"]]
        print(f"  [{tag}] decode solo (1 in flight): " + " ".join(f"{d:.1f}" for d in dsolo) + " tok/s")

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
