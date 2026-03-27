"""
Example: Delta DOS plot using vasprunrs as a drop-in Vasprun replacement.

Adapted from no_bi2se3_compare_delta.py. The only import-level change is
swapping pymatgen's Vasprun for Vasprunrs — everything else (CompleteDos,
Spin, Element, Locpot, etc.) comes from pymatgen as usual.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── swap this one line ────────────────────────────────────────────────────────
# from pymatgen.io.vasp import Vasprun          # original pymatgen (slow)
from vasprunrs.pymatgen import Vasprunrs as Vasprun   # fast Rust-backed shim
# ─────────────────────────────────────────────────────────────────────────────

from pymatgen.io.vasp.outputs import Locpot
from pymatgen.electronic_structure.dos import CompleteDos, Spin
from pymatgen.core import Element

# ----------------------------
# User inputs
# ----------------------------
bare_dirs = {
    "AuBi2Te3": "AuBi2Te3/soc",
    "AgBi2Te3": "AgBi2Te3/soc",
    "PtBi2Te3": "PtBi2Te3/soc",
}

bound_dirs = {
    "CO-AuBi2Te3": "COAuBi2Te3/soc",
    "CO-AgBi2Te3": "COAgBi2Te3/soc",
    "H-PtBi2Te3":  "HPtBi2Te3/soc",
}

vacuum_correction = True

# LOCPOT paths for vacuum potential alignment (None = no correction for that pair)
locpot_paths = {
    "AuBi2Te3":    "AuBi2Te3/soc/LOCPOT",
    "CO-AuBi2Te3": "COAuBi2Te3/soc/LOCPOT",
    "AgBi2Te3":    None,
    "CO-AgBi2Te3": None,
    "PtBi2Te3":    None,
    "H-PtBi2Te3":  None,
}


def get_vacuum_level(locpot_path):
    """
    Return the vacuum electrostatic potential (eV) from a LOCPOT file.
    Uses the planar average along z; the vacuum level is estimated as the
    mean of values in the top-5th-percentile plateau.
    """
    locpot = Locpot.from_file(locpot_path)
    avg = locpot.get_average_along_axis(2)
    threshold = np.percentile(avg, 95)
    return float(np.mean(avg[avg >= threshold]))


sigma = 0.2
emin, emax = -16, 2

# Colors for plotting
color_map = {
    "Metal":     "#9E9E9E",   # Ag, Au, Pt
    "Bi":        "#7B1FA2",
    "Te":        "#2E7D32",
    "Adsorbate": "#ba8759",
}

legend_groups = {
    "Metal":     ["Ag", "Au", "Pt"],
    "Bi":        ["Bi"],
    "Te":        ["Te"],
    "Adsorbate": ["H", "C", "O"],
}

# ----------------------------
# Font sizes
# ----------------------------
mpl.rcParams.update({
    "font.size":       16,
    "axes.titlesize":  16,
    "axes.labelsize":  16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ----------------------------
# Plot helpers
# ----------------------------
def get_element_delta_dos(bare_dos, bound_dos, element, e_grid, e_shift=0.0):
    """
    Interpolates element DOS from bare and bound onto a common grid and
    returns the difference.  If the element is absent in the bare slab
    (e.g. adsorbate), its bare DOS is treated as zero.

    e_shift: vacuum potential correction (eV) added to the bound energy axis
             so that both systems share a common vacuum reference.
             e_shift = (V_vac_bare - E_F_bare) - (V_vac_bound - E_F_bound)
    """
    element_dos_map_bare  = bare_dos.get_element_dos()
    element_dos_map_bound = bound_dos.get_element_dos()

    if element in element_dos_map_bare:
        bare_energies  = bare_dos.energies - bare_dos.efermi
        bare_densities = element_dos_map_bare[element].get_smeared_densities(sigma)[Spin.up]
        bare_d = np.interp(e_grid, bare_energies, bare_densities)
    else:
        bare_d = np.zeros_like(e_grid)

    if element in element_dos_map_bound:
        # Apply vacuum correction: shift bound energy axis by e_shift
        bound_energies  = bound_dos.energies - bound_dos.efermi + e_shift
        bound_densities = element_dos_map_bound[element].get_smeared_densities(sigma)[Spin.up]
        bound_d = np.interp(e_grid, bound_energies, bound_densities)
    else:
        bound_d = np.zeros_like(e_grid)

    return bound_d - bare_d


def plot_delta_dos(ax, delta, e_grid, color="#AAAAAA", alpha=0.6):
    """Plots a delta DOS with positive and negative filled regions."""
    patch_pos = ax.fill_betweenx(e_grid, 0, delta, where=(delta >= 0),
                                 color=color, alpha=alpha)
    ax.fill_betweenx(e_grid, 0, delta, where=(delta < 0),
                     color=color, alpha=alpha * 0.4)
    ax.plot(delta, e_grid, color=color, linewidth=1.5)
    return patch_pos


# ----------------------------
# Main plotting
# ----------------------------
system_pairs = list(zip(bare_dirs.items(), bound_dirs.items()))

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(system_pairs),
    figsize=(16, 6),
    sharey=True,
)

if len(system_pairs) == 1:
    axes = [axes]

e_grid = np.linspace(emin, emax, 2000)

for ax, ((bare_label, bare_path), (bound_label, bound_path)) in zip(axes, system_pairs):
    # Vasprunrs accepts the same keyword arguments as pymatgen's Vasprun.
    bare_vasprun = Vasprun(
        os.path.join(bare_path, "vasprun.xml"),
        parse_potcar_file=False,
        parse_projected_eigen=True,
    )
    bound_vasprun = Vasprun(
        os.path.join(bound_path, "vasprun.xml"),
        parse_potcar_file=False,
        parse_projected_eigen=True,
    )

    # get_complete_dos() is the Vasprunrs equivalent of pymatgen's .complete_dos
    bare_dos:  CompleteDos = bare_vasprun.get_complete_dos()
    bound_dos: CompleteDos = bound_vasprun.get_complete_dos()

    # Vacuum potential correction: align both systems to the same vacuum reference
    e_shift = 0.0
    if vacuum_correction:
        bare_lp  = locpot_paths.get(bare_label)
        bound_lp = locpot_paths.get(bound_label)
        if bare_lp and bound_lp:
            v_vac_bare  = get_vacuum_level(bare_lp)
            v_vac_bound = get_vacuum_level(bound_lp)
            e_shift = (v_vac_bare - bare_dos.efermi) - (v_vac_bound - bound_dos.efermi)
            print(f"{bare_label} / {bound_label}: V_vac_bare={v_vac_bare:.4f} eV, "
                  f"V_vac_bound={v_vac_bound:.4f} eV, E_shift={e_shift:.4f} eV")

    elements_in_bound = {site.specie.symbol for site in bound_vasprun.final_structure}

    handles = {}

    for group_name, group_elements in legend_groups.items():
        for element_symbol in group_elements:
            if element_symbol in elements_in_bound:
                delta = get_element_delta_dos(
                    bare_dos, bound_dos, Element(element_symbol), e_grid, e_shift=e_shift
                )
                patch = plot_delta_dos(ax, delta, e_grid, color=color_map[group_name])
                if group_name not in handles:
                    handles[group_name] = patch

    ax.axhline(0, color="k", linestyle="--", linewidth=1)    # Fermi level
    ax.axvline(0, color="k", linestyle="-",  linewidth=0.8)  # zero DOS reference
    ax.set_ylim(emin, emax)
    ax.set_title(f"Δ {bound_label}")
    ax.set_xlabel("ΔDOS")

axes[-1].legend(handles.values(), handles.keys(), loc="lower right")
axes[0].set_ylabel("Energy (eV)")

plt.tight_layout()
plt.savefig("delta_ldos.png")
plt.show()
