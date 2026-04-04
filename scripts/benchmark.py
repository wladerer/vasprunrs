"""
Benchmark: vasprunrs (Rust) vs pymatgen (Python)

Usage:
    python scripts/benchmark.py [<path/to/vasprun.xml[.gz]> ...] [--runs N]

If no paths are given, the test fixtures are used.
pymatgen does not support .gz natively; those files are decompressed to a
temporary file for the pymatgen timing only.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import statistics
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("xml", nargs="*", help="vasprun.xml(.gz) paths to benchmark")
parser.add_argument("--runs", type=int, default=5, help="parse repetitions (default: 5)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_FILES = [
    REPO_ROOT / "tests/si_rpa.xml.gz",
    REPO_ROOT / "tests/mos2_soc.xml.gz",
    REPO_ROOT / "tests/n_gw_large.xml.gz",
]

files = [Path(p) for p in args.xml] if args.xml else [p for p in DEFAULT_FILES if p.exists()]
RUNS = args.runs


def timeit(fn, runs):
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, times


def fmt(times):
    return (
        f"mean={statistics.mean(times):8.1f} ms  "
        f"median={statistics.median(times):8.1f} ms  "
        f"min={min(times):7.1f} ms"
    )


def file_size_str(path: Path) -> str:
    mb = path.stat().st_size / 1e6
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{mb * 1000:.0f} KB"


def decompress_to_tmp(path: Path) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    with gzip.open(path, "rb") as src:
        shutil.copyfileobj(src, tmp)
    tmp.close()
    return tmp.name


import vasprunrs  # noqa: E402
from vasprunrs.pymatgen import Vasprunrs  # noqa: E402

try:
    import warnings

    from pymatgen.io.vasp.outputs import Vasprun as PmgVasprun
    HAS_PMG = True
except ImportError:
    HAS_PMG = False

results = []

for xml in files:
    size = file_size_str(xml)
    label = xml.stem.replace(".xml", "")
    print(f"\n{'─'*70}")
    print(f"  {label}  [{size}]")
    print(f"{'─'*70}")

    vr, times_rs = timeit(lambda p=str(xml): vasprunrs.Vasprun(p), RUNS)
    print(f"  vasprunrs   {fmt(times_rs)}")

    _, times_shim = timeit(lambda p=str(xml): Vasprunrs(p), RUNS)
    print(f"  Vasprunrs   {fmt(times_shim)}")

    eig_shape = vr.eigenvalue_shape
    dos = vr.dos
    diel = vr.dielectric
    print(
        f"  → atoms={len(vr.atoms)}  kpts={vr.kpoints['kpointlist'].shape[0]}"
        f"  steps={len(vr.ionic_steps)}"
        f"  eigs={eig_shape}"
        f"  dos={'✓' if dos else '✗'}"
        f"  diel={'✓' if diel is not None else '✗'}"
    )

    mean_shim = statistics.mean(times_shim)
    if HAS_PMG:
        plain = decompress_to_tmp(xml) if str(xml).endswith(".gz") else str(xml)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, times_pm = timeit(
                    lambda p=plain: PmgVasprun(p, parse_projected_eigen=False), RUNS
                )
            mean_pm = statistics.mean(times_pm)
            speedup = mean_pm / statistics.mean(times_rs)
            speedup_s = mean_pm / mean_shim
            print(f"  pymatgen    {fmt(times_pm)}")
            print(f"  → speedup (raw): {speedup:.1f}×  |  speedup (shim): {speedup_s:.1f}×")
            results.append((label, size, statistics.mean(times_rs), mean_shim, mean_pm))
        finally:
            if plain != str(xml):
                Path(plain).unlink(missing_ok=True)
    else:
        results.append((label, size, statistics.mean(times_rs), mean_shim, None))

if len(results) > 1:
    print(f"\n{'═'*70}")
    print(f"  {'File':<30} {'Size':>7}  {'raw':>8}  {'shim':>8}  {'pymatgen':>8}  {'speedup':>8}")
    print(f"  {'-'*30} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for lbl, sz, tr, ts, tp in results:
        sp = f"{tp/tr:.1f}×" if tp is not None else "N/A"
        pm_col = f"{tp:6.0f} ms" if tp is not None else "     N/A"
        print(f"  {lbl:<30} {sz:>7}  {tr:6.0f} ms  {ts:6.0f} ms  {pm_col}  {sp:>8}")
    print(f"{'═'*70}\n")
