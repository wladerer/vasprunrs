"""
Benchmark and showcase: vasprunrs (Rust) vs pymatgen (Python)

Usage:
    python benchmark.py [<path/to/vasprun.xml> ...] [--runs N]

If no paths are given, a default set of test files is used.
"""

import sys
import os
import time
import argparse
import statistics
from pathlib import Path

# ── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("xml", nargs="*", help="vasprun.xml paths to benchmark")
parser.add_argument("--runs", type=int, default=5,
                    help="Number of parse repetitions (default: 5)")
args = parser.parse_args()

DEFAULT_FILES = [
    "tests/si_rpa.xml",
    "tests/mos2_soc.xml",
    "tests/n_gw_large.xml",
    "/home/wladerer/research/interfaces/dosses/AuBi2Te3/band/vasprun.xml",
    "/home/wladerer/research/interfaces/dosses/AuBi2Te3/soc/vasprun.xml",
    "/home/wladerer/research/interfaces/dosses/AgBi2Te3/soc/vasprun.xml",
    "/home/wladerer/research/interfaces/dosses/COAgBi2Te3/soc/vasprun.xml",
]

files = args.xml if args.xml else [f for f in DEFAULT_FILES if Path(f).exists()]
RUNS = args.runs

# ── helpers ──────────────────────────────────────────────────────────────────
def timeit(fn, runs):
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, times

def stats(times):
    return dict(mean=statistics.mean(times), median=statistics.median(times),
                min=min(times), max=max(times))

def fmt(times):
    s = stats(times)
    return (f"mean={s['mean']:8.1f} ms  "
            f"median={s['median']:8.1f} ms  "
            f"min={s['min']:7.1f} ms")

def file_size_str(path):
    mb = Path(path).stat().st_size / 1e6
    if mb >= 1000:
        return f"{mb/1000:.1f} GB"
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{mb*1000:.0f} KB"

# ── imports ──────────────────────────────────────────────────────────────────
import vasprunrs
from vasprunrs.pymatgen import Vasprunrs

try:
    from pymatgen.io.vasp.outputs import Vasprun as PmgVasprun
    import warnings
    HAS_PMG = True
except ImportError:
    HAS_PMG = False

# ── run ──────────────────────────────────────────────────────────────────────
results = []

for xml in files:
    size = file_size_str(xml)
    parts = Path(xml).parts
    label = "/".join(parts[-3:-1]) if len(parts) >= 3 else Path(xml).stem
    print(f"\n{'─'*70}")
    print(f"  {label}  [{size}]")
    print(f"{'─'*70}")

    # vasprunrs (raw Rust)
    vr, times_rs = timeit(lambda p=xml: vasprunrs.Vasprun(p), RUNS)
    print(f"  vasprunrs   {fmt(times_rs)}")

    # Vasprunrs (pymatgen shim)
    _, times_shim = timeit(lambda p=xml: Vasprunrs(p), RUNS)
    print(f"  Vasprunrs   {fmt(times_shim)}")

    # Summarise what was parsed
    eig_shape = vr.eigenvalue_shape
    dos = vr.dos
    diel = vr.dielectric
    print(f"  → atoms={vr.atoms}  kpts={vr.kpoints['kpointlist'].shape[0]}"
          f"  steps={len(vr.ionic_steps)}"
          f"  eigs={eig_shape}"
          f"  dos={'✓' if dos else '✗'}"
          f"  diel={'✓' if diel is not None else '✗'}")

    # pymatgen
    mean_shim = statistics.mean(times_shim)
    if HAS_PMG:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, times_pm = timeit(lambda p=xml: PmgVasprun(p, parse_projected_eigen=False), RUNS)
        mean_pm   = statistics.mean(times_pm)
        speedup   = mean_pm / statistics.mean(times_rs)
        speedup_s = mean_pm / mean_shim
        print(f"  pymatgen    {fmt(times_pm)}")
        print(f"  → speedup (raw):  {speedup:.1f}×  |  speedup (shim): {speedup_s:.1f}×")
        results.append((label, size, statistics.mean(times_rs), mean_shim, mean_pm))
    else:
        results.append((label, size, statistics.mean(times_rs), mean_shim, None))

# ── summary table ─────────────────────────────────────────────────────────────
if len(results) > 1:
    print(f"\n{'═'*70}")
    print(f"  {'File':<30} {'Size':>7}  {'raw':>8}  {'shim':>8}  {'pymatgen':>8}  {'speedup':>8}")
    print(f"  {'-'*30} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for (lbl, sz, tr, ts, tp) in results:
        if tp is not None:
            sp = f"{tp/tr:.1f}×"
        else:
            sp = "N/A"
        pm_col = f"{tp:6.0f} ms" if tp is not None else "     N/A"
        print(f"  {lbl:<30} {sz:>7}  {tr:6.0f} ms  {ts:6.0f} ms  {pm_col}  {sp:>8}")
    print(f"{'═'*70}\n")
