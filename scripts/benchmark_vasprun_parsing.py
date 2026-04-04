"""
Benchmark: stdlib xml.etree.ElementTree vs lxml + tag-filtered iterparse
for pymatgen Vasprun parsing.

Compares:
  1. Raw iterparse overhead (event counts + tokenization time)
  2. Full Vasprun.parse() time with the current lxml + tag-filter implementation

Results from development benchmarking (separate machine, pre-PR baseline):
  vasprun.xml.gz  (13 MB): stdlib ~517 ms → lxml+filter ~158 ms  (3.3×)
  n_gw_large.xml  (20 MB): stdlib ~1117 ms → lxml+filter ~399 ms  (2.8×)

Usage:
    uv run python benchmark_vasprun_parsing.py [file1.xml[.gz] ...]
"""
from __future__ import annotations

import gzip
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path
from xml.etree import ElementTree as _stdlib_ET

from lxml import etree as _lxml_ET

_TAGS = [
    "atominfo", "calculation", "dielectricfunction", "dos", "dynmat",
    "eigenvalues", "eigenvalues_kpoints_opt", "energy", "generator",
    "incar", "kpoints", "parameters", "projected", "projected_kpoints_opt",
    "structure", "varray",
]

DEFAULT_FILES = [
    "test-files/io/vasp/outputs/vasprun.xml.gz",
    "test-files/io/vasp/outputs/vasprun.dfpt.ionic.xml.gz",
    "test-files/io/vasp/outputs/vasprun.int_Te_SOC.xml.gz",
]


def _decompress(xml_path: str) -> tuple[str, int]:
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    if xml_path.endswith(".gz"):
        with gzip.open(xml_path, "rb") as src:
            shutil.copyfileobj(src, tmp)
    else:
        with open(xml_path, "rb") as src:
            shutil.copyfileobj(src, tmp)
    size = tmp.tell()
    tmp.close()
    return tmp.name, size


def _iterparse_bench(xml_path: str, use_lxml: bool, tag_filter: bool, runs: int = 3) -> tuple[int, float]:
    """Return (event_count, mean_ms). No Python logic, pure event dispatch cost."""
    ET = _lxml_ET if use_lxml else _stdlib_ET
    kwargs: dict = {"events": ["start", "end"]}
    if use_lxml and tag_filter:
        kwargs["tag"] = _TAGS

    times = []
    n = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        n = 0
        for _evt, elem in ET.iterparse(xml_path, **kwargs):
            n += 1
            if _evt == "end":
                elem.clear()
        times.append((time.perf_counter() - t0) * 1000)
    return n, statistics.mean(times)


def _vasprun_bench(xml_path: str, runs: int = 5, use_stdlib: bool = False) -> list[float]:
    """Time full Vasprun parse. use_stdlib patches in a stdlib-compatible ET wrapper."""
    import pymatgen.io.vasp.outputs as _mod
    from pymatgen.io.vasp.outputs import Vasprun

    if use_stdlib:
        # Wrap stdlib ET so it accepts but ignores lxml's tag= kwarg
        class _StdlibCompat:
            @staticmethod
            def iterparse(source, events=None, tag=None):
                return _stdlib_ET.iterparse(source, events=events)
            XMLSyntaxError = _stdlib_ET.ParseError
            ParseError = _stdlib_ET.ParseError
        _mod.ET = _StdlibCompat()  # type: ignore[assignment]

    try:
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Vasprun(xml_path, parse_projected_eigen=False)
            times.append((time.perf_counter() - t0) * 1000)
        return times
    finally:
        if use_stdlib:
            _mod.ET = _lxml_ET  # type: ignore[assignment]


def fmt_times(times: list[float]) -> str:
    return (f"mean={statistics.mean(times):7.0f} ms  "
            f"min={min(times):6.0f} ms  "
            f"median={statistics.median(times):6.0f} ms")


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in DEFAULT_FILES if Path(f).exists()]
    if not files:
        print("No test files found. Pass paths as arguments.")
        sys.exit(1)

    RUNS = 5
    summary = []

    for xml_path in files:
        tmp_path, raw_bytes = _decompress(xml_path)
        size_mb = raw_bytes / 1e6
        size_str = f"{size_mb:.0f} MB" if size_mb >= 1 else f"{int(size_mb * 1000)} KB"
        name = Path(xml_path).name

        print(f"\n{'─'*70}")
        print(f"  {name}  [{size_str} uncompressed]")
        print(f"{'─'*70}")

        n_std,  t_std  = _iterparse_bench(tmp_path, use_lxml=False, tag_filter=False)
        n_lxml, t_lxml = _iterparse_bench(tmp_path, use_lxml=True,  tag_filter=False)
        n_filt, t_filt = _iterparse_bench(tmp_path, use_lxml=True,  tag_filter=True)

        reduction = n_std // max(n_filt, 1)
        print("\n  Raw iterparse overhead (no Python logic, 3 runs each):")
        print(f"  stdlib  (no filter):  {n_std:>9,} events  {t_std:5.0f} ms")
        print(f"  lxml    (no filter):  {n_lxml:>9,} events  {t_lxml:5.0f} ms  ({t_std/t_lxml:.1f}× faster than stdlib)")
        print(f"  lxml  (tag filter):   {n_filt:>9,} events  {t_filt:5.0f} ms  ({reduction:,}× fewer callbacks)")

        t_stdlib = _vasprun_bench(tmp_path, runs=RUNS, use_stdlib=True)
        t_lxml_full = _vasprun_bench(tmp_path, runs=RUNS, use_stdlib=False)
        speedup = statistics.mean(t_stdlib) / statistics.mean(t_lxml_full)

        print(f"\n  Full Vasprun parse — {RUNS} runs:")
        print(f"  stdlib  (baseline):    {fmt_times(t_stdlib)}")
        print(f"  lxml  (tag filter):    {fmt_times(t_lxml_full)}  ← {speedup:.1f}× faster")

        Path(tmp_path).unlink()
        summary.append((name, size_str, statistics.mean(t_stdlib), statistics.mean(t_lxml_full), speedup, n_std, n_filt, reduction))

    print(f"\n{'═'*70}")
    print(f"  {'File':<30} {'Size':>6}  {'stdlib':>8}  {'lxml+filter':>11}  {'speedup':>7}")
    print(f"  {'-'*30} {'-'*6}  {'-'*8}  {'-'*11}  {'-'*7}")
    for name, size, t_std, t_opt, sp, n_std, n_filt, reduction in summary:
        print(f"  {name:<30} {size:>6}  {t_std:6.0f} ms  {t_opt:9.0f} ms  {sp:6.1f}×")
    print(f"{'═'*70}")
