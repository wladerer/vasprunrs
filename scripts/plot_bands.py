"""
Projected band structure plot from vasprunrs .npz output.

Usage
-----
    # export first:
    vasprunrs bands vasprun.xml -o bands.npz --projected --shift-efermi

    # plain bands (no projection):
    python scripts/plot_bands.py bands.npz

    # fat bands colored by orbital group:
    python scripts/plot_bands.py bands.npz --orbital d --spin 0

    # multiple orbitals on one plot (different colors):
    python scripts/plot_bands.py bands.npz --orbital s p d

    # save without displaying:
    python scripts/plot_bands.py bands.npz --orbital d -o fatbands.png

.npz schema (produced by `vasprunrs bands --projected`)
-------------------------------------------------------
    eigenvalues : (nspins, nkpts, nbands)   energies in eV
    occupancies : (nspins, nkpts, nbands)
    kpoints     : (nkpts, 3)                fractional reciprocal coords
    weights     : (nkpts,)
    efermi      : scalar
    labels      : object array of (kpt_index, label) pairs  [optional]
    projected   : (nspins, nkpts, nbands, nions, norbitals) [optional]
    orbitals    : object array of orbital label strings      [optional]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Orbital groupings (standard VASP ordering)
# ---------------------------------------------------------------------------

ORBITAL_GROUPS: dict[str, list[str]] = {
    "s":  ["s"],
    "p":  ["px", "py", "pz"],
    "d":  ["dxy", "dyz", "dz2", "dxz", "dx2", "x2-y2", "dx2-y2"],
    "f":  ["f-3", "f-2", "f-1", "f0", "f1", "f2", "f3"],
    "sp": ["s", "px", "py", "pz"],
    "pd": ["px", "py", "pz", "dxy", "dyz", "dz2", "dxz", "dx2"],
}

COLORS = ["#e06c75", "#61afef", "#98c379", "#e5c07b", "#c678dd", "#56b6c2"]


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def kpath_distances(kpoints: np.ndarray) -> np.ndarray:
    """Cumulative distances along the k-path in reciprocal fractional units."""
    deltas = np.diff(kpoints, axis=0)
    dists  = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(dists)])


def detect_segment_breaks(kpoints: np.ndarray, tol: float = 1e-6) -> list[int]:
    """
    Indices where the k-path jumps discontinuously (line-mode band structures).
    Returns indices of the *first* point of each new segment (excluding 0).
    """
    breaks = []
    for i in range(1, len(kpoints) - 1):
        d_prev = np.linalg.norm(kpoints[i] - kpoints[i - 1])
        d_next = np.linalg.norm(kpoints[i + 1] - kpoints[i])
        if d_prev > 1e-4 and d_next < tol:
            # kpoints[i] == kpoints[i+1]: start of new segment
            breaks.append(i + 1)
    return breaks


def parse_labels(labels_arr) -> dict[int, str]:
    """Convert the object array of (index, label) pairs to a dict."""
    result = {}
    for item in labels_arr:
        try:
            idx, lbl = int(item[0]), str(item[1])
            result[idx] = lbl
        except (TypeError, IndexError, ValueError):
            pass
    return result


def orbital_weight(
    projected: np.ndarray,
    orbitals: list[str],
    orbital_group: str,
    spin: int,
) -> np.ndarray:
    """
    Sum projected weights for a named orbital group.

    Parameters
    ----------
    projected : (nspins, nkpts, nbands, nions, norbitals)
    orbitals  : list of orbital label strings from the .npz
    orbital_group : key in ORBITAL_GROUPS or a bare orbital name

    Returns
    -------
    weight : (nkpts, nbands)  summed over ions and selected orbitals
    """
    names = ORBITAL_GROUPS.get(orbital_group, [orbital_group])
    indices = [i for i, o in enumerate(orbitals) if o in names]
    if not indices:
        print(f"  Warning: no orbitals matched '{orbital_group}' in {orbitals}", file=sys.stderr)
        nkpts, nbands = projected.shape[1], projected.shape[2]
        return np.zeros((nkpts, nbands))
    # sum over ions (axis=3) and selected orbitals
    w = projected[spin, :, :, :, indices]  # (nkpts, nbands, nions, len(indices))
    return w.sum(axis=(2, 3))              # (nkpts, nbands)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bands(
    data: dict,
    orbital_groups: list[str],
    spin: int,
    emin: float,
    emax: float,
    output: str | None,
    lw: float,
    scale: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for plotting:  pip install matplotlib")

    eigs   = data["eigenvalues"]           # (nspins, nkpts, nbands)
    kpts   = data["kpoints"]               # (nkpts, 3)
    labels = parse_labels(data.get("labels", np.array([], dtype=object)))

    nspins, nkpts, nbands = eigs.shape
    if spin >= nspins:
        sys.exit(f"spin={spin} requested but file only has {nspins} spin channel(s)")

    kdist  = kpath_distances(kpts)
    breaks = detect_segment_breaks(kpts)

    projected  = data.get("projected")    # may be None
    orb_labels = list(data["orbitals"]) if "orbitals" in data else []

    fig, ax = plt.subplots(figsize=(7, 5))

    # -- draw each band --
    for b in range(nbands):
        ene = eigs[spin, :, b]

        if not orbital_groups or projected is None:
            # plain band structure
            ax.plot(kdist, ene, color="#555555", lw=lw, rasterized=True)
        else:
            # fat bands: one pass per orbital group
            for g, (grp, color) in enumerate(zip(orbital_groups, COLORS)):
                w = orbital_weight(projected, orb_labels, grp, spin)[:, b].ravel()
                # clip weights to non-negative
                w = np.clip(w, 0, None)
                first_seg = True
                for seg_start, seg_end in _segments(nkpts, breaks):
                    if seg_end <= seg_start:
                        continue
                    x  = kdist[seg_start:seg_end]
                    y  = ene[seg_start:seg_end]
                    ww = w[seg_start:seg_end]
                    ax.plot(x, y, color="#cccccc", lw=lw * 0.5, zorder=1, rasterized=True)
                    # scatter: size encodes orbital weight
                    ax.scatter(x, y, s=ww * scale, color=color, linewidths=0,
                               zorder=2 + g,
                               label=(grp if (b == 0 and first_seg) else None),
                               rasterized=True)
                    first_seg = False

    # -- segment break lines --
    for br in breaks:
        ax.axvline(kdist[br], color="#aaaaaa", lw=0.8, ls="--")

    # -- high-symmetry labels --
    tick_pos, tick_lbl = _make_ticks(kdist, labels, breaks, nkpts)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)
    ax.tick_params(axis="x", length=0)
    for tp in tick_pos:
        ax.axvline(tp, color="#aaaaaa", lw=0.8)

    # -- Fermi level reference --
    ax.axhline(0.0, color="#aaaaaa", lw=0.8, ls=":")

    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(emin, emax)
    ax.set_ylabel("Energy (eV)")
    spin_label = "" if nspins == 1 else f"  [spin {'up' if spin == 0 else 'dn'}]"
    ax.set_title(f"Band structure{spin_label}")

    if orbital_groups and projected is not None:
        handles = [
            plt.scatter([], [], s=80, color=COLORS[i], label=g)
            for i, g in enumerate(orbital_groups)
        ]
        ax.legend(handles=handles, title="Orbital", loc="upper right",
                  framealpha=0.9, fontsize=9)

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved {output}")
    else:
        plt.show()


def _segments(nkpts: int, breaks: list[int]) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each continuous k-path segment."""
    edges = [0] + breaks + [nkpts]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _make_ticks(
    kdist: np.ndarray,
    labels: dict[int, str],
    breaks: list[int],
    nkpts: int,
) -> tuple[list[float], list[str]]:
    """Build tick positions and labels, merging coincident segment boundaries."""
    # Collect all candidate tick positions
    candidate_indices = sorted({0, nkpts - 1} | set(labels) | set(breaks) |
                                {b - 1 for b in breaks})
    candidate_indices = [i for i in candidate_indices if 0 <= i < nkpts]

    positions: list[float] = []
    tick_labels: list[str] = []
    for idx in candidate_indices:
        pos = float(kdist[idx])
        lbl = labels.get(idx, "")
        # If previous tick is at same position, merge labels with |
        if positions and abs(positions[-1] - pos) < 1e-8:
            prev = tick_labels[-1]
            tick_labels[-1] = f"{prev}|{lbl}" if lbl else prev
        else:
            positions.append(pos)
            tick_labels.append(lbl)

    return positions, tick_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot (projected) band structure from vasprunrs .npz output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("npz", help="Path to the .npz file (from vasprunrs bands)")
    parser.add_argument("--orbital", "-r", nargs="+", metavar="GROUP", default=[],
                        help="Orbital group(s) to project onto: s p d f sp pd "
                             "(or any orbital name). Omit for plain band structure.")
    parser.add_argument("--spin",    "-s", type=int, default=0, metavar="N",
                        help="Spin channel index: 0=up (default), 1=down")
    parser.add_argument("--emin",    type=float, default=-6.0,
                        help="Lower energy limit in eV relative to Fermi (default: -6)")
    parser.add_argument("--emax",    type=float, default=6.0,
                        help="Upper energy limit in eV relative to Fermi (default: +6)")
    parser.add_argument("--scale",   type=float, default=200.0,
                        help="Dot-size scale factor for fat bands (default: 200)")
    parser.add_argument("--lw",      type=float, default=0.6,
                        help="Base line width for bands (default: 0.6)")
    parser.add_argument("-o", "--output", default=None, metavar="FILE",
                        help="Save to file instead of displaying (e.g. bands.png)")
    args = parser.parse_args()

    if not Path(args.npz).exists():
        sys.exit(f"File not found: {args.npz}")

    data = load(args.npz)

    if args.orbital and "projected" not in data:
        print("Warning: --orbital requested but no projected data in .npz.", file=sys.stderr)
        print("Re-export with:  vasprunrs bands vasprun.xml -o bands.npz --projected",
              file=sys.stderr)

    print(f"Loaded {args.npz}")
    print(f"  eigenvalues : {data['eigenvalues'].shape}  (nspins, nkpts, nbands)")
    if "projected" in data:
        print(f"  projected   : {data['projected'].shape}")
        print(f"  orbitals    : {list(data['orbitals'])}")
    print(f"  efermi      : {float(data['efermi']):.4f} eV")
    print(f"  labels      : {parse_labels(data.get('labels', np.array([], dtype=object)))}")

    plot_bands(
        data=data,
        orbital_groups=args.orbital,
        spin=args.spin,
        emin=args.emin,
        emax=args.emax,
        output=args.output,
        lw=args.lw,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
