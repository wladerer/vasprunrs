"""
Command-line interface for vasprunrs.

Usage:
    vasprunrs [file] [options]    -- convergence summary (default: ./vasprun.xml)
    vasprunrs bands <file> -o x.npz
    vasprunrs dos   <file> -o x.npz
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import click
import numpy as np

from .vasprunrs import Vasprun as _RustVasprun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str, parse_dos: bool = False, parse_eigen: bool = False,
          parse_projected: bool = False) -> _RustVasprun:
    try:
        return _RustVasprun(path, parse_dos=parse_dos, parse_eigen=parse_eigen,
                            parse_projected=parse_projected)
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


def _fmt_de(val: float | None) -> str:
    if val is None:
        return "--"
    return f"{val:+.3e}"


def _fmt_lgde(val: float | None) -> str:
    if val is None or val == 0.0:
        return "--"
    try:
        return f"{math.log10(abs(val)):+.2f}"
    except ValueError:
        return "--"


def _force_norm(f: list) -> float:
    return math.sqrt(sum(x * x for x in f))


def _fmax(forces: list, selective: list | None) -> float:
    norms = [
        _force_norm(f) for i, f in enumerate(forces)
        if selective is None or any(selective[i])
    ]
    return max(norms) if norms else 0.0


def _favg(forces: list, selective: list | None) -> float:
    norms = [
        _force_norm(f) for i, f in enumerate(forces)
        if selective is None or any(selective[i])
    ]
    return sum(norms) / len(norms) if norms else 0.0


def _fmax_info(forces: list, selective: list | None) -> tuple[float, int, str]:
    best_norm, best_idx, best_ax = -1.0, 0, "X"
    for i, f in enumerate(forces):
        if selective is not None and not any(selective[i]):
            continue
        n = _force_norm(f)
        if n > best_norm:
            best_norm = n
            best_idx = i
            best_ax = "XYZ"[max(range(3), key=lambda j: abs(f[j]))]
    return best_norm, best_idx + 1, best_ax


def _frozen_count(selective: list | None) -> int:
    if selective is None:
        return 0
    return sum(1 for flags in selective if not any(flags))


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
# Display
# ---------------------------------------------------------------------------

def _print_header(vr: _RustVasprun, fname: str) -> None:
    ver  = vr.version
    prog = vr.program
    ncore = _incar_int(vr, "NCORE", 1)
    npar  = _incar_int(vr, "NPAR",  1)
    kpar  = _incar_int(vr, "KPAR",  1)
    atoms = vr.atoms
    species_counts: dict[str, int] = {}
    for a in atoms:
        species_counts[a] = species_counts.get(a, 0) + 1
    species_str = "  ".join(f"{v} x {k}" for k, v in species_counts.items())

    click.echo(f"\n  {fname}")
    click.echo(f"  {'-' * len(fname)}")
    click.echo(f"  VASP {ver}  {prog}")
    click.echo(f"  NCORE={ncore}  NPAR={npar}  KPAR={kpar}")
    click.echo(f"  Atoms: {species_str}    K-points: {vr.kpoints['kpointlist'].shape[0]}")


def _print_scf_table(scf: list, nelm: int, ediff: float, label: str = "Electronic convergence") -> None:
    if not scf:
        click.echo(f"\n  {label}: no SCF steps recorded")
        return
    w_scf  = 4
    w_e    = 16
    w_de   = 12
    w_lg   = 8
    click.echo(f"\n  {label}")
    click.echo(f"  {'SCF':>{w_scf}}  {'E_wo_entrp (eV)':>{w_e}}  {'dE (eV)':>{w_de}}  {'log|dE|':>{w_lg}}")
    click.echo(f"  {'-'*w_scf}  {'-'*w_e}  {'-'*w_de}  {'-'*w_lg}")
    prev = None
    for i, s in enumerate(scf):
        e  = s["e_wo_entrp"]
        de = (e - prev) if prev is not None else None
        prev = e
        flag = "  [!]" if (i + 1) == nelm else ""
        click.echo(
            f"  {i+1:>{w_scf}}  {e:>{w_e}.6f}  {_fmt_de(de):>{w_de}}  {_fmt_lgde(de):>{w_lg}}{flag}"
        )
    last_de = scf[-1]["e_wo_entrp"] - scf[-2]["e_wo_entrp"] if len(scf) > 1 else None
    if last_de is not None:
        if abs(last_de) < ediff:
            click.echo(f"\n  Result: CONVERGED  (|dE| {abs(last_de):.1e} < EDIFF {ediff:.1e})")
        else:
            click.echo(f"\n  Result: NOT CONVERGED  (|dE| {abs(last_de):.1e}  EDIFF {ediff:.1e})")


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
    nsw    = _incar_int(vr, "NSW",    0)
    nelm   = _incar_int(vr, "NELM",   60)
    ediff  = _incar_float(vr, "EDIFF",  1e-4)
    ediffg = _incar_float(vr, "EDIFFG", ediff * 10)
    isif   = _incar_int(vr, "ISIF",   2)
    ispin  = _incar_int(vr, "ISPIN",  1)

    is_single_point = (ibrion == -1 or nsw == 0)

    click.echo()
    click.echo("  Calculation")
    click.echo(f"  Type   : {_ibrion_label(ibrion)}")
    if not is_single_point:
        click.echo(f"  Cell   : {_isif_label(isif)}")
        click.echo(f"  Max    : NSW={nsw} ionic steps,  NELM={nelm} SCF/step")
    else:
        click.echo(f"  Max SCF: NELM={nelm}")
    click.echo(f"  EDIFF  : {ediff:.1e} eV  (electronic convergence)")
    if not is_single_point:
        sign = "force" if ediffg < 0 else "energy"
        unit = "eV/A" if ediffg < 0 else "eV"
        click.echo(f"  EDIFFG : {ediffg:.3g} {unit}  ({sign} convergence)")

    steps = vr.ionic_steps
    if not steps:
        click.echo("\n  No ionic steps parsed.")
        return

    if is_single_point:
        _print_scf_table(steps[0].get("scf_steps", []), nelm, ediff)
        return

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


def _print_ionic_table(
    steps, nelm, ediff, ediffg, isif, ispin,
    print_toten, print_favg, print_fmax_axis, print_fmax_index,
    print_volume, no_fmax, no_lgde, no_magmom, no_nscf,
) -> None:
    use_forces   = ediffg < 0
    force_target = abs(ediffg) if use_forces else None

    # Column definitions: (key, header, width)
    col_defs = [
        ("#",               "#",              5),
        ("e_wo_entrp",      "E_wo_entrp (eV)", 16),
    ]
    if print_toten:
        col_defs.append(("e_fr",       "E_fr (eV)",    14))
    col_defs.append(("de",          "dE (eV)",      12))
    if not no_lgde:
        col_defs.append(("lgde",      "log|dE|",       8))
    if not no_fmax:
        col_defs.append(("fmax",      "Fmax (eV/A)",  11))
    if print_favg:
        col_defs.append(("favg",      "Favg (eV/A)",  11))
    if print_fmax_axis:
        col_defs.append(("fax",       "F-ax",          4))
    if print_fmax_index:
        col_defs.append(("fidx",      "F-idx",         5))
    if isif >= 2 or print_volume:
        col_defs.append(("vol",       "Vol (A^3)",    10))
    if ispin == 2 and not no_magmom:
        col_defs.append(("mag",       "magmom (uB)",  11))
    if not no_nscf:
        col_defs.append(("nscf",      "nSCF",          5))

    keys   = [c[0] for c in col_defs]
    heads  = [c[1] for c in col_defs]
    widths = [c[2] for c in col_defs]

    def row_str(cells: list[str]) -> str:
        return "  " + "  ".join(v.ljust(w) for v, w in zip(cells, widths))

    click.echo()
    click.echo("  Ionic convergence")
    click.echo(row_str(heads))
    click.echo(row_str(["-" * w for w in widths]))

    frozen_warned = False
    lgde_vals: list[float] = []
    prev_e = None

    for idx, step in enumerate(steps):
        e    = step["e_wo_entrp"]
        e_fr = step["e_fr_energy"]
        de   = (e - prev_e) if prev_e is not None else None
        prev_e = e
        if de is not None:
            lgde_vals.append(math.log10(abs(de)) if de != 0.0 else -99.0)

        sel    = step["structure"].get("selective")
        forces = step.get("forces", [])
        fm, fidx, fax = _fmax_info(forces, sel) if forces else (0.0, 0, "X")
        fa     = _favg(forces, sel) if forces else 0.0
        vol    = step["structure"]["volume"]
        scf    = step.get("scf_steps", [])
        nscf   = len(scf)
        mag    = _magmom_total(step)

        # Step number: append flag character for notable events (no spaces, fits in width)
        step_flag = "+" if (de is not None and de > 0) else " "
        nscf_flag = "!" if nscf == nelm else " "
        fmax_flag = "*" if (use_forces and force_target is not None and fm <= force_target) else " "

        cells: list[str] = []
        for key in keys:
            if key == "#":
                cells.append(f"{idx + 1}{step_flag}")
            elif key == "e_wo_entrp":
                cells.append(f"{e:+.6f}")
            elif key == "e_fr":
                cells.append(f"{e_fr:+.6f}")
            elif key == "de":
                cells.append(_fmt_de(de))
            elif key == "lgde":
                cells.append(_fmt_lgde(de))
            elif key == "fmax":
                cells.append(f"{fm:.4f}{fmax_flag}{nscf_flag}")
            elif key == "favg":
                cells.append(f"{fa:.4f}")
            elif key == "fax":
                cells.append(fax)
            elif key == "fidx":
                cells.append(str(fidx))
            elif key == "vol":
                cells.append(f"{vol:.3f}")
            elif key == "mag":
                cells.append(f"{mag:.3f}" if mag is not None else "--")
            elif key == "nscf":
                cells.append(f"{nscf}{nscf_flag}")

        click.echo(row_str(cells))

        if sel and not frozen_warned and _frozen_count(sel) > 0:
            frozen_warned = True

    # SCF detail for the last ionic step
    last_scf = steps[-1].get("scf_steps", [])
    _print_scf_table(last_scf, nelm, ediff, label="Last ionic step -- electronic convergence")

    # Frozen atom warning
    if frozen_warned:
        nfrozen = _frozen_count(steps[-1]["structure"].get("selective"))
        click.echo(f"\n  [!] {nfrozen} atom(s) frozen -- forces on frozen atoms are real but excluded from Fmax")

    # Pathological warnings
    if lgde_vals:
        for i, v in enumerate(lgde_vals):
            if v > 0:
                click.echo(f"\n  [+] Energy increased at ionic step {i + 2}")
                break
        signs = [1 if v > 0 else -1 for v in lgde_vals]
        if len(signs) > 2 and all(signs[i] != signs[i + 1] for i in range(len(signs) - 1)):
            click.echo("\n  [!] Oscillating convergence detected")

    # Result
    last   = steps[-1]
    sel    = last["structure"].get("selective")
    forces = last.get("forces", [])
    fm     = _fmax(forces, sel) if forces else 0.0

    click.echo()
    if use_forces:
        if fm <= abs(ediffg):
            click.echo(f"  Result: CONVERGED  (Fmax {fm:.4f} <= |EDIFFG| {abs(ediffg):.4f} eV/A)")
        else:
            click.echo(f"  Result: NOT CONVERGED  (Fmax {fm:.4f} > |EDIFFG| {abs(ediffg):.4f} eV/A)")
    else:
        last_de = lgde_vals[-1] if lgde_vals else None
        if last_de is not None and 10 ** last_de <= ediffg:
            click.echo(f"  Result: CONVERGED  (|dE| {10**last_de:.1e} <= EDIFFG {ediffg:.1e} eV)")
        else:
            click.echo("  Result: NOT CONVERGED")

    trend = _convergence_trend(lgde_vals)
    if trend:
        click.echo(f"  Trend : {trend}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

class _DefaultToInfo(click.Group):
    """Group that forwards unrecognised first arguments to the `info` command."""

    def resolve_command(self, ctx: click.Context, args: list) -> tuple:
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name not in self.commands and not cmd_name.startswith("-"):
            args.insert(0, "info")
        return super().resolve_command(ctx, args)


@click.group(cls=_DefaultToInfo, invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Fast VASP vasprun.xml inspector."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(info)


@main.command()
@click.argument("file", default="./vasprun.xml", required=False)
@click.option("-e", "--toten",   is_flag=True, help="Also print TOTEN (free energy with entropy)")
@click.option("-a", "--favg",    is_flag=True, help="Print average force magnitude")
@click.option("-x", "--fmaxis",  is_flag=True, help="Print axis of max force component")
@click.option("-i", "--fmidx",   is_flag=True, help="Print index of atom with max force (1-based)")
@click.option("-v", "--volume",  is_flag=True, help="Print cell volume per ionic step")
@click.option("--no-fmax",   is_flag=True, help="Suppress Fmax column")
@click.option("--no-lgde",   is_flag=True, help="Suppress log|dE| column")
@click.option("--no-magmom", is_flag=True, help="Suppress magnetic moment column")
@click.option("--no-nscf",   is_flag=True, help="Suppress SCF count column")
@click.option("--watch", "watch_interval", default=0, type=float, metavar="SEC",
              help="Poll file every SEC seconds (0 = run once)")
def info(file, toten, favg, fmaxis, fmidx, volume,
         no_fmax, no_lgde, no_magmom, no_nscf, watch_interval):
    """Convergence summary. Defaults to ./vasprun.xml."""

    def _run_once() -> None:
        vr = _load(file)
        _print_header(vr, Path(file).name)
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

    if not watch_interval:
        _run_once()
        return

    click.echo(f"Watching {file}  (every {watch_interval}s, Ctrl-C to stop)")
    last_mtime = None
    while True:
        try:
            mtime = Path(file).stat().st_mtime
        except FileNotFoundError:
            click.echo(f"  Waiting for {file} ...")
            time.sleep(watch_interval)
            continue

        if mtime != last_mtime:
            last_mtime = mtime
            click.clear()
            click.echo(f"  updated {time.strftime('%H:%M:%S')}")
            try:
                _run_once()
            except Exception as exc:
                click.echo(f"  [parse error] {exc}")
        time.sleep(watch_interval)


@main.command()
@click.argument("file", default="./vasprun.xml", required=False)
@click.option("-o", "--output", required=True, help="Output .npz file")
@click.option("--shift-efermi", is_flag=True, help="Subtract Fermi energy from eigenvalues")
@click.option("--projected",    is_flag=True, help="Include projected eigenvalues (large)")
def bands(file, output, shift_efermi, projected):
    """Export eigenvalues and k-points to a .npz file."""
    vr = _load(file, parse_eigen=True, parse_projected=projected)
    eigs = vr.eigenvalues
    if eigs is None:
        raise click.ClickException("No eigenvalue data in this file")

    efermi      = vr.efermi or 0.0
    kpts        = vr.kpoints
    energies    = eigs[..., 0] - (efermi if shift_efermi else 0.0)
    occupancies = eigs[..., 1]

    save_kwargs: dict = dict(
        eigenvalues=energies,
        occupancies=occupancies,
        kpoints=kpts["kpointlist"],
        weights=np.array(kpts["weights"]),
        efermi=np.float64(efermi),
        labels=np.array(kpts.get("labels", []), dtype=object),
    )

    if projected:
        p = vr.projected
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
@click.argument("file", default="./vasprun.xml", required=False)
@click.option("-o", "--output", required=True, help="Output .npz file")
def dos(file, output):
    """Export DOS to a .npz file."""
    vr = _load(file, parse_dos=True)
    d  = vr.dos
    if d is None:
        raise click.ClickException("No DOS data in this file (LORBIT tag required)")

    save_kwargs: dict = dict(
        energies=np.array(d["total"]["energies"]),
        total=d["total"]["densities"],
        integrated=d["total"]["integrated"],
        efermi=np.float64(d["efermi"]),
    )
    if d.get("partial") is not None:
        save_kwargs["partial"]  = d["partial"]["data"]
        save_kwargs["orbitals"] = np.array(d["partial"]["orbitals"], dtype=object)

    np.savez(output, **save_kwargs)
    click.echo(f"  Saved {output}")
    click.echo(f"  Keys : {list(save_kwargs.keys())}")
