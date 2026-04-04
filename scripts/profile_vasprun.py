"""
Generate CPU and memory profiles for stdlib ET vs lxml + tag-filter Vasprun parsing.

Usage:
    # CPU flamegraph (via py-spy, run as two separate commands):
    py-spy record -o profile_stdlib.svg -- python profile_vasprun.py stdlib <file>
    py-spy record -o profile_lxml.svg   -- python profile_vasprun.py lxml   <file>

    # Memory profile (via memray):
    memray run -o memray_stdlib.bin profile_vasprun.py stdlib <file>
    memray run -o memray_lxml.bin   profile_vasprun.py lxml   <file>
    memray flamegraph memray_stdlib.bin -o memray_stdlib.html
    memray flamegraph memray_lxml.bin   -o memray_lxml.html

    # Or just run directly for a quick tracemalloc peak memory comparison:
    python profile_vasprun.py compare <file>
"""
from __future__ import annotations

import sys
import tracemalloc
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


def run_stdlib(xml_path: str) -> None:
    import pymatgen.io.vasp.outputs as _mod
    _orig = _mod.ET

    class _StdlibCompat:
        @staticmethod
        def iterparse(source, events=None, tag=None):
            return _stdlib_ET.iterparse(source, events=events)
        XMLSyntaxError = _stdlib_ET.ParseError
        ParseError = _stdlib_ET.ParseError

    _mod.ET = _StdlibCompat()  # type: ignore
    try:
        from pymatgen.io.vasp.outputs import Vasprun
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Vasprun(xml_path, parse_projected_eigen=False)
    finally:
        _mod.ET = _orig  # type: ignore


def run_lxml(xml_path: str) -> None:
    from pymatgen.io.vasp.outputs import Vasprun
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Vasprun(xml_path, parse_projected_eigen=False)


def compare_memory(xml_path: str) -> None:
    print(f"\nPeak memory comparison — {Path(xml_path).name}")
    print(f"{'─'*50}")

    tracemalloc.start()
    run_stdlib(xml_path)
    _, peak_stdlib = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    run_lxml(xml_path)
    _, peak_lxml = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def fmt_mb(b): return f"{b / 1e6:.1f} MB"
    print(f"  stdlib  peak memory: {fmt_mb(peak_stdlib)}")
    print(f"  lxml    peak memory: {fmt_mb(peak_lxml)}")
    print(f"  reduction: {peak_stdlib / peak_lxml:.2f}×")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mode, xml_path = sys.argv[1], sys.argv[2]

    if mode == "stdlib":
        run_stdlib(xml_path)
    elif mode == "lxml":
        run_lxml(xml_path)
    elif mode == "compare":
        compare_memory(xml_path)
    else:
        print(f"Unknown mode: {mode}. Use stdlib, lxml, or compare.")
        sys.exit(1)
