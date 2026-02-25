# Benchmark logic (boot_bench)

## Goal

One robust threshold per config: **"If you have more than N files (or size Y), use parallel; otherwise sequential."**  
No millisecond precision.

---

## Constants (boot_bench.py)

| Constant | Value | Meaning |
|----------|--------|--------|
| `N_START_BY_MP` | 0.3 → 1000, 5 → 200 | First n (files) to test; start high, then step down |
| `N_MIN` | 5 | Lower bound for n (for very large MP) |
| `MAX_IMAGES_PER_CONFIG` | 2500 | Cap images generated per config |
| `PAR_MIN_FASTER_RATIO` | 0.95 | Par wins only if `t_par < t_seq * 0.95` (≥5% faster); else seq (overhead) |
| `CLOSE_ENOUGH_RATIO` | 0.9 | Stop when at crossover seq and par are within 10% |
| `N_RUNS_PER_MEASURE` | 3 | Runs per (n, seq/par); interleaved order; take **median** |

---

## Per-config loop (_run_one_config)

**Initial n:** `n = max(N_MIN, min(n_start, n_max, n_max//2))`  
So for 0.3 MP: n_start=1000 → first n is 1000 (or n_max if smaller). For 5 MP: first n = 200 (or less if n_max &lt; 200).

**For each n (until break or no next n):**

1. **Measure**
   - 3 runs, **interleaved** (run 1: seq then par, run 2: par then seq, run 3: seq then par).
   - `t_seq` = median of 3 seq times, `t_par` = median of 3 par times.

2. **Winner**
   - `par_wins = (t_par < t_seq * 0.95)`  
   - `winner = "par"` only if par is ≥5% faster; else `"seq"`.

3. **If par_wins**
   - Update `best_n`, `best_size`.
   - If we already have at least one **seq** point and **times within 10%** → **break** (done).
   - If we have at least one seq point and ≥2 metrics:
     - Compute **crossover** n between largest seq-n and smallest par-n (linear interpolation).
     - If that n is between them and not yet tested → **next n = crossover**, continue.
     - Else → no bisection; fall through.
   - If we have **no seq point yet** → **step down**: `n_next = n_min + (n - n_min)//2`; if valid and not tested, use it and continue.
   - Else → **break**.

4. **If not par_wins (seq wins)**
   - **Next n** from `_next_n_adaptive`: go **up** (e.g. n*2, n*1.5, or n*5 depending on how much seq won).  
   - If that returns `None` (par would win meaningfully) → loop ends (no next n).

**Result:** `best_n` / `best_size` = last n where par won (with 5% rule); saved as threshold for that config.

---

## Flow diagram (one config)

```
n = n_start (e.g. 1000 for 0.3 MP)
while n valid and n not tested:
  measure seq/par at n (3 runs, median)
  winner = par only if t_par < t_seq * 0.95

  if par_wins:
    best_n = n
    if has_seq and within 10% → STOP
    if has_seq and ≥2 points:
      n_cross = interpolate(seq_pt, par_pt)
      if n_cross in (left, right) and not tested → n = n_cross; continue
    if no seq yet:
      n = n_down (bisect toward N_MIN); continue
    → STOP

  else (seq wins):
    n = next_n_adaptive (step up)
    if n is None → STOP
```

---

## Phase 1 vs Phase 2

- **Phase 1:** Generate images (0.3MP_8b, 0.3MP_16b, 5MP_8b, 5MP_16b, etc.) in a temp dir; max `MAX_IMAGES_PER_CONFIG` per config, `n_max` also limited by `max_raw_gb`.
- **Phase 2:** For each config, run `_run_one_config` (seq/par sweep above). Results → `thresholds_per_config[label] = { files: best_n, size_bytes: best_size }`, plus `fallback_files` / `fallback_size` (max over configs). These are written to settings and `boot_bench_results.json` (+ history in `boot_bench_results/`).

---

## Where it can "get stuck"

- **No seq point:** If par wins at every n we try (e.g. we only step down and never see seq), we break with `best_n` set but no crossover refinement. So we still get a threshold, but the curve may only have par-winning points until we hit n_min.
- **Crossover already tested:** If the interpolated `n_final` was already in `tested`, we don’t run again at crossover and break (no bisection). So we can end with no extra point exactly at crossover.
- **Adaptive returns None:** When seq wins but par is almost as fast (`t_par < t_seq * 0.95`), `_next_n_adaptive` returns `None` and the loop exits. So we don’t step up further even though we only have seq points.

If you want to "unstick", typical levers: lower `PAR_MIN_FASTER_RATIO` (e.g. 0.98) so par wins more easily, or ensure `n_start` is in a range where you get one seq and one par point before hitting the 10% or crossover logic.
