"""
Command-line interface for vasprunrs.

Usage:
    vasprunrs info  <file>              -- convergence summary
    vasprunrs stats <file>              -- parallelism and environment
    vasprunrs watch <file> [-n SEC]     -- live monitor (polls file)
    vasprunrs bands <file> -o out.npz   -- export eigenvalues
    vasprunrs dos   <file> -o out.npz   -- export DOS
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np

from .vasprunrs import Vasprun as _RustVasprun

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str, parse_dos: bool = False, parse_eigen: bool = False) -> _RustVasprun:
    try:
        return _RustVasprun(path, parse_dos=parse_dos, parse_eigen=parse_eigen)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def _incar_int(vr: _RustVasprun, key: str, default: int) -> int:
    val = vr.incar.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _incar_float(vr: _RustVasprun, key: str, default: float) -> float:
    val = vr.incar.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_e(val: float) -> str:
    return f"{val:+.6f}"


def _fmt_de(val: float | None) -> str:
    if val is None:
        return "        --"
    return f"{val:+.3e}"


def _fmt_lgde(val: float | None) -> str:
    if val is None or val == 0.0:
        return "   --"
    try:
        return f"{math.log10(abs(val)):+.2f}"
    except ValueError:
        return "   --"


def _force_norm(f: list) -> float:
    return math.sqrt(sum(x * x for x in f))


def _fmax(forces: list, selective: list | None) -> float:
    norms = []
    for i, f in enumerate(forces):
        if selective is not None:
            flags = selective[i]
            if not any(flags):
                continue
        norms.append(_force_norm(f))
    return max(norms) if norms else 0.0


def _favg(forces: list, selective: list | None) -> float:
    norms = []
    for i, f in enumerate(forces):
        if selective is not None:
            flags = selective[i]
            if not any(flags):
                continue
        norms.append(_force_norm(f))
    return sum(norms) / len(norms) if norms else 0.0


def _fmax_info(forces: list, selective: list | None) -> tuple[float, int, str]:
    best_norm = -1.0
    best_idx = 0
    for i, f in enumerate(forces):
        if selective is not None and not any(selective[i]):
            continue
        n = _force_norm(f)
        if n > best_norm:
            best_norm = n
            best_idx = i
            ax = max(range(3), key=lambda j: abs(f[j]))
            best_ax = "XYZ"[ax]
    return best_norm, best_idx + 1, best_ax


def _frozen_count(selective: list | None) -> int:
    if selective is None:
        return 0
    return sum(1 for flags in selective if not any(flags))


def _scf_lgde_row(scf_steps: list) -> str:
    if len(scf_steps) < 2:
        return "--"
    vals = []
    for i in range(1, len(scf_steps)):
        de = scf_steps[i]["e_wo_entrp"] - scf_steps[i - 1]["e_wo_entrp"]
        vals.append(_fmt_lgde(de).strip())
    return " ".join(vals[:12])  # cap at 12 entries to avoid line overflow


def _magmom_total(step: dict) -> float | None:
    mag = step.get("magnetization")
    if not mag:
        return None
    return sum(m[0] if len(m) >= 1 else 0.0 for m in mag)


def _ibrion_label(ibrion: int) -> str:
    labels = {
        -1: "Single point",
        0:  "Molecular dynamics",
        1:  "Geometry optimization (RMM-DIIS)",
        2:  "Geometry optimization (CG)",
        3:  "Geometry optimization (damped MD)",
        5:  "Finite differences (phonons)",
        6:  "Perturbation theory (phonons)",
        44: "Transition state (dimer)",
        40: "Intrinsic reaction coordinate",
    }
    return labels.get(ibrion, f"IBRION={ibrion}")


def _isif_label(isif: int) -> str:
    labels = {
        0: "Forces only, no stress",
        1: "Forces + stress trace",
        2: "Forces + full stress",
        3: "Relax ions + shape + volume",
        4: "Relax ions + shape",
        5: "Relax shape only",
        6: "Relax shape + volume",
        7: "Relax volume only",
    }
    return labels.get(isif, f"ISIF={isif}")


def _convergence_trend(lgde_vals: list[float]) -> str | None:
    if len(lgde_vals) < 2:
        return None
    diffs = [lgde_vals[i] - lgde_vals[i - 1] for i in range(1, len(lgde_vals))]
    rate = sum(diffs) / len(diffs)
    return f"{rate:+.2f} log-units/step"


# ---------------------------------------------------------------------------
# Stats section (shared between info and stats commands)
# ---------------------------------------------------------------------------

def _print_stats(vr: _RustVasprun, filename: str) -> None:
    gen = vr.program
    ver = vr.version
    ncore = _incar_int(vr, "NCORE", 1)
    npar  = _incar_int(vr, "NPAR", 1)
    kpar  = _incar_int(vr, "KPAR", 1)

    click.echo(f"  VASP {ver}  {gen}")
    click.echo(f"  NCORE={ncore}  NPAR={npar}  KPAR={kpar}")

    atoms = vr.atoms
    species_counts: dict[str, int] = {}
    for a in atoms:
        species_counts[a] = species_counts.get(a, 0) + 1
    species_str = "  ".join(f"{v} x {k}" for k, v in species_counts.items())
    click.echo(f"  Atoms    : {species_str}")
    click.echo(f"  K-points : {vr.kpoints['kpointlist'].shape[0]}")


# ---------------------------------------------------------------------------
# Convergence section
# ---------------------------------------------------------------------------

def _print_convergence(
    vr: _RustVasprun,
    *,
    print_toten: bool,
    print_favg: bool,
    print_fmax_axis: bool,
    print_fmax_index: bool,
    print_volume: bool,
    no_fmax: bool,
    no_lgde: bool,
    no_magmom: bool,
    no_nscf: bool,
) -> None:
    ibrion = _incar_int(vr, "IBRION", -1)
    nsw    = _incar_int(vr, "NSW", 0)
    nelm   = _incar_int(vr, "NELM", 60)
    ediff  = _incar_float(vr, "EDIFF", 1e-4)
    ediffg = _incar_float(vr, "EDIFFG", ediff * 10)
    isif   = _incar_int(vr, "ISIF", 2)
    ispin  = _incar_int(vr, "ISPIN", 1)

    is_single_point = (ibrion == -1 or nsw == 0)

    click.echo()
    click.echo("  Calculation")
    click.echo(f"  Type    : {_ibrion_label(ibrion)}")
    if not is_single_point:
        click.echo(f"  Cell    : {_isif_label(isif)}")
        click.echo(f"  Max     : NSW={nsw} ionic steps,  NELM={nelm} SCF/step")
    else:
        click.echo(f"  Max SCF : NELM={nelm}")
    click.echo(f"  EDIFF   : {ediff:.1e} eV  (electronic convergence)")
    if not is_single_point:
        sign = "force" if ediffg < 0 else "energy"
        unit = "eV/A" if ediffg < 0 else "eV"
        click.echo(f"  EDIFFG  : {ediffg:.3g} {unit}  ({sign} convergence)")

    steps = vr.ionic_steps
    if not steps:
        click.echo("\n  No ionic steps parsed.")
        return

    # Single-point: show SCF table for the one step
    if is_single_point:
        _print_scf_table(steps[0], nelm, ediff)
        return

    # Geometry optimization: show ionic table
    _print_ionic_table(
        steps, nelm, ediff, ediffg, isif, ispin,
        print_toten=print_toten,
        print_favg=print_favg,
        print_fmax_axis=print_fmax_axis,
        print_fmax_index=print_fmax_index,
        print_volume=print_volume,
        no_fmax=no_fmax,
        no_lgde=no_lgde,
        no_magmom=no_magmom,
        no_nscf=no_nscf,
    )


def _print_scf_table(step: dict, nelm: int, ediff: float) -> None:
    scf = step.get("scf_steps", [])
    click.echo()
    click.echo("  Electronic convergence")
    click.echo(f"  {'SCF':>4}  {'E_wo_entrp (eV)':>18}  {'dE (eV)':>12}  {'log|dE|':>8}")
    click.echo(f"  {'-'*4}  {'-'*18}  {'-'*12}  {'-'*8}")
    prev = None
    for i, s in enumerate(scf):
        e = s["e_wo_entrp"]
        de = (e - prev) if prev is not None else None
        prev = e
        flag = " [!]" if (i + 1) == nelm else ""
        click.echo(f"  {i+1:>4}  {e:>18.6f}  {_fmt_de(de):>12}  {_fmt_lgde(de):>8}{flag}")
    if scf:
        last_de = scf[-1]["e_wo_entrp"] - scf[-2]["e_wo_entrp"] if len(scf) > 1 else None
        if last_de is not None and abs(last_de) < ediff:
            click.echo(f"\n  Result : CONVERGED electronically  (|dE| {abs(last_de):.1e} < EDIFF {ediff:.1e})")
        else:
            click.echo(f"\n  Result : NOT CONVERGED  (last |dE| {abs(last_de):.1e} vs EDIFF {ediff:.1e})")


def _print_ionic_table(
    steps, nelm, ediff, ediffg, isif, ispin,
    print_toten, print_favg, print_fmax_axis, print_fmax_index,
    print_volume, no_fmax, no_lgde, no_magmom, no_nscf,
) -> None:
    use_forces = ediffg < 0
    force_target = abs(ediffg) if use_forces else None

    # Build header
    cols = ["#", "E_wo_entrp (eV)"]
    if print_toten:
        cols.append("E_fr (eV)")
    cols.append("dE (eV)")
    if not no_lgde:
        cols.append("log|dE|")
    if not no_fmax:
        cols.append("Fmax (eV/A)")
    if print_favg:
        cols.append("Favg (eV/A)")
    if print_fmax_axis:
        cols.append("F-ax")
    if print_fmax_index:
        cols.append("F-idx")
    if isif >= 2 or print_volume:
        cols.append("Vol (A^3)")
    if ispin == 2 and not no_magmom:
        cols.append("magmom (uB)")
    if not no_nscf:
        cols.append("nSCF")
    cols.append("SCF log|dE|")

    widths = {
        "#": 4,
        "E_wo_entrp (eV)": 16,
        "E_fr (eV)": 14,
        "dE (eV)": 12,
        "log|dE|": 8,
        "Fmax (eV/A)": 11,
        "Favg (eV/A)": 11,
        "F-ax": 4,
        "F-idx": 5,
        "Vol (A^3)": 10,
        "magmom (uB)": 11,
        "nSCF": 5,
        "SCF log|dE|": 30,
    }

    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  " + "  ".join("-" * widths[c] for c in cols)
    click.echo()
    click.echo("  Ionic convergence")
    click.echo(header)
    click.echo(sep)

    frozen_warned = False
    lgde_vals: list[float] = []
    prev_e = None

    for idx, step in enumerate(steps):
        e = step["e_wo_entrp"]
        e_fr = step["e_fr_energy"]
        de = (e - prev_e) if prev_e is not None else None
        prev_e = e
        if de is not None:
            lgde_vals.append(math.log10(abs(de)) if de != 0.0 else -99.0)

        sel = step["structure"].get("selective")
        forces = step.get("forces", [])
        fm, fidx, fax = _fmax_info(forces, sel) if forces else (0.0, 0, "X")
        fa = _favg(forces, sel) if forces else 0.0
        vol = step["structure"]["volume"]
        scf = step.get("scf_steps", [])
        nscf = len(scf)
        mag = _magmom_total(step)
        scf_row = _scf_lgde_row(scf)

        # convergence flags
        nscf_flag = " [!]" if nscf == nelm else ""
        de_flag = ""
        if de is not None and de > 0:
            de_flag = " [+]"

        fmax_str = f"{fm:.4f}"
        if use_forces and force_target is not None and fm <= force_target:
            fmax_str += " *"

        row_vals: list[str] = []
        row_vals.append(str(idx + 1).rjust(widths["#"]))
        row_vals.append(f"{e:+.6f}".ljust(widths["E_wo_entrp (eV)"]))
        if print_toten:
            row_vals.append(f"{e_fr:+.6f}".ljust(widths["E_fr (eV)"]))
        de_str = (_fmt_de(de) + de_flag).ljust(widths["dE (eV)"])
        row_vals.append(de_str)
        if not no_lgde:
            row_vals.append(_fmt_lgde(de).ljust(widths["log|dE|"]))
        if not no_fmax:
            row_vals.append((fmax_str + nscf_flag).ljust(widths["Fmax (eV/A)"]))
        if print_favg:
            row_vals.append(f"{fa:.4f}".ljust(widths["Favg (eV/A)"]))
        if print_fmax_axis:
            row_vals.append(fax.ljust(widths["F-ax"]))
        if print_fmax_index:
            row_vals.append(str(fidx).ljust(widths["F-idx"]))
        if isif >= 2 or print_volume:
            row_vals.append(f"{vol:.3f}".ljust(widths["Vol (A^3)"]))
        if ispin == 2 and not no_magmom:
            mag_str = f"{mag:.3f}" if mag is not None else "--"
            row_vals.append(mag_str.ljust(widths["magmom (uB)"]))
        if not no_nscf:
            row_vals.append(str(nscf).ljust(widths["nSCF"]))
        row_vals.append(scf_row)

        click.echo("  " + "  ".join(row_vals))

        if sel and not frozen_warned:
            nfrozen = _frozen_count(sel)
            if nfrozen > 0:
                frozen_warned = True

    if frozen_warned:
        nfrozen = _frozen_count(steps[-1]["structure"].get("selective"))
        click.echo(f"\n  [!] {nfrozen} atom(s) frozen -- forces on frozen atoms are real but excluded from Fmax")

    # Warnings
    if lgde_vals:
        for i, v in enumerate(lgde_vals):
            if v > 0:
                click.echo(f"\n  [+] Energy increased at ionic step {i + 2} -- check geometry")
                break
        signs = [1 if v > 0 else -1 for v in lgde_vals]
        alternating = all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
        if alternating and len(signs) > 2:
            click.echo("\n  [!] Oscillating convergence detected")

    # Result
    last = steps[-1]
    sel = last["structure"].get("selective")
    forces = last.get("forces", [])
    fm = _fmax(forces, sel) if forces else 0.0

    click.echo()
    if use_forces:
        if fm <= abs(ediffg):
            click.echo(f"  Result : CONVERGED  (Fmax {fm:.4f} <= |EDIFFG| {abs(ediffg):.4f} eV/A)")
        else:
            click.echo(f"  Result : NOT CONVERGED  (Fmax {fm:.4f} > |EDIFFG| {abs(ediffg):.4f} eV/A)")
    else:
        last_de = lgde_vals[-1] if lgde_vals else None
        if last_de is not None and 10 ** last_de <= ediffg:
            click.echo(f"  Result : CONVERGED  (|dE| {10**last_de:.1e} <= EDIFFG {ediffg:.1e} eV)")
        else:
            click.echo("  Result : NOT CONVERGED")

    trend = _convergence_trend(lgde_vals)
    if trend:
        click.echo(f"  Trend  : {trend}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """Fast VASP vasprun.xml inspector."""


@main.command()
@click.argument("file")
@click.option("-e", "--toten",    is_flag=True, help="Also print TOTEN (free energy with entropy)")
@click.option("-a", "--favg",     is_flag=True, help="Print average force magnitude")
@click.option("-x", "--fmaxis",   is_flag=True, help="Print axis of max force component")
@click.option("-i", "--fmidx",    is_flag=True, help="Print index of atom with max force (1-based)")
@click.option("-v", "--volume",   is_flag=True, help="Print cell volume per step")
@click.option("--no-fmax",    is_flag=True, help="Suppress Fmax column")
@click.option("--no-lgde",    is_flag=True, help="Suppress log|dE| column")
@click.option("--no-magmom",  is_flag=True, help="Suppress magnetic moment column")
@click.option("--no-nscf",    is_flag=True, help="Suppress SCF count column")
def info(file, toten, favg, fmaxis, fmidx, volume, no_fmax, no_lgde, no_magmom, no_nscf):
    """Convergence summary for a finished or in-progress calculation."""
    vr = _load(file)
    fname = Path(file).name
    click.echo(f"\n  {fname}")
    click.echo(f"  {'-' * len(fname)}")
    _print_stats(vr, fname)
    _print_convergence(
        vr,
        print_toten=toten,
        print_favg=favg,
        print_fmax_axis=fmaxis,
        print_fmax_index=fmidx,
        print_volume=volume,
        no_fmax=no_fmax,
        no_lgde=no_lgde,
        no_magmom=no_magmom,
        no_nscf=no_nscf,
    )
    click.echo()


@main.command()
@click.argument("file")
def stats(file):
    """Environment and parallelism summary."""
    vr = _load(file)
    fname = Path(file).name
    click.echo(f"\n  Stats -- {fname}")
    _print_stats(vr, fname)
    click.echo()


@main.command()
@click.argument("file")
@click.option("-n", "--interval", default=5.0, show_default=True, help="Polling interval in seconds")
@click.option("-e", "--toten",    is_flag=True)
@click.option("--no-fmax",    is_flag=True)
@click.option("--no-lgde",    is_flag=True)
@click.option("--no-magmom",  is_flag=True)
@click.option("--no-nscf",    is_flag=True)
def watch(file, interval, toten, no_fmax, no_lgde, no_magmom, no_nscf):
    """Poll a vasprun.xml and refresh the convergence display."""
    click.echo(f"Watching {file}  (interval {interval}s, Ctrl-C to stop)")
    last_mtime = None
    while True:
        try:
            mtime = Path(file).stat().st_mtime
        except FileNotFoundError:
            click.echo(f"  Waiting for {file} ...")
            time.sleep(interval)
            continue

        if mtime != last_mtime:
            last_mtime = mtime
            click.clear()
            click.echo(f"  {file}  (updated {time.strftime('%H:%M:%S')})")
            try:
                vr = _load(file)
                _print_stats(vr, Path(file).name)
                _print_convergence(
                    vr,
                    print_toten=toten,
                    print_favg=False,
                    print_fmax_axis=False,
                    print_fmax_index=False,
                    print_volume=False,
                    no_fmax=no_fmax,
                    no_lgde=no_lgde,
                    no_magmom=no_magmom,
                    no_nscf=no_nscf,
                )
            except Exception as exc:
                click.echo(f"  [parse error] {exc}")
        time.sleep(interval)


@main.command()
@click.argument("file")
@click.option("-o", "--output", required=True, help="Output .npz file")
@click.option("--shift-efermi", is_flag=True, help="Subtract Fermi energy from eigenvalues")
@click.option("--projected",    is_flag=True, help="Include projected eigenvalues (large)")
def bands(file, output, shift_efermi, projected):
    """Export eigenvalues and k-points to a .npz file."""
    vr = _load(file, parse_eigen=True)
    eigs = vr.eigenvalues
    if eigs is None:
        raise click.ClickException("No eigenvalue data in this vasprun.xml (parse_eigen=True required)")

    efermi = vr.efermi or 0.0
    kpts = vr.kpoints
    energies   = eigs[..., 0]
    occupancies = eigs[..., 1]
    if shift_efermi:
        energies = energies - efermi

    save_kwargs: dict = dict(
        eigenvalues=energies,
        occupancies=occupancies,
        kpoints=kpts["kpointlist"],
        weights=np.array(kpts["weights"]),
        efermi=np.float64(efermi),
        labels=np.array(kpts.get("labels", []), dtype=object),
    )

    if projected:
        vr2 = _load(file, parse_eigen=True, parse_projected=True)
        p = vr2.projected
        if p is not None:
            save_kwargs["projected"] = p["data"]
            save_kwargs["orbitals"]  = np.array(p["orbitals"], dtype=object)
        else:
            click.echo("  [!] No projected eigenvalues found in file")

    np.savez(output, **save_kwargs)
    click.echo(f"  Saved {output}")
    click.echo(f"  Keys : {list(save_kwargs.keys())}")
    click.echo(f"  Shape: eigenvalues {energies.shape}")


@main.command()
@click.argument("file")
@click.option("-o", "--output", required=True, help="Output .npz file")
def dos(file, output):
    """Export DOS to a .npz file."""
    vr = _load(file, parse_dos=True)
    d = vr.dos
    if d is None:
        raise click.ClickException("No DOS data in this vasprun.xml (LORBIT tag required)")

    save_kwargs: dict = dict(
        energies=np.array(d["total"]["energies"]),
        total=d["total"]["densities"],
        integrated=d["total"]["integrated"],
        efermi=np.float64(d["efermi"]),
    )
    if "partial" in d and d["partial"] is not None:
        save_kwargs["partial"]  = d["partial"]["data"]
        save_kwargs["orbitals"] = np.array(d["partial"]["orbitals"], dtype=object)

    np.savez(output, **save_kwargs)
    click.echo(f"  Saved {output}")
    click.echo(f"  Keys : {list(save_kwargs.keys())}")
