"""
pymatgen compatibility shim for vasprunrs.

Provides ``Vasprunrs``, a drop-in replacement for
``pymatgen.io.vasp.outputs.Vasprun`` backed by the fast Rust parser.

Usage::

    from vasprunrs.pymatgen import Vasprunrs

    v = Vasprunrs("vasprun.xml")
    structure  = v.final_structure      # pymatgen Structure
    bs         = v.get_band_structure() # BandStructure / BandStructureSymmLine
    dos        = v.get_complete_dos()   # CompleteDos

Only pymatgen objects are returned from properties — the raw Rust data is
accessed via ``v._raw`` if needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .vasprunrs import Vasprun as _RustVasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.electronic_structure.bandstructure import (
        BandStructure,
        BandStructureSymmLine,
    )
    from pymatgen.electronic_structure.dos import CompleteDos, Dos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pmg_structure(raw_dict: dict) -> "Structure":
    from pymatgen.core import Structure, Lattice
    lattice = Lattice(raw_dict["lattice"])
    return Structure(
        lattice=lattice,
        species=raw_dict["species"],
        coords=raw_dict["positions"],
        coords_are_cartesian=False,
    )


# VASP orbital label → pymatgen Orbital enum key
_VASP_ORBITAL_MAP: dict[str, str] = {
    "x2-y2":  "dx2",
    "dx2-y2": "dx2",
}


def _pmg_spin(nspins: int):
    from pymatgen.electronic_structure.core import Spin
    if nspins == 2:
        return [Spin.up, Spin.down]
    return [Spin.up]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Vasprunrs:
    """
    Drop-in replacement for ``pymatgen.io.vasp.outputs.Vasprun``.

    Parameters
    ----------
    filename:
        Path to vasprun.xml.
    parse_dos:
        Parse the DOS block (default True).
    parse_eigen:
        Parse eigenvalues (default True).
    parse_projected_eigen:
        Parse projected eigenvalues — large, off by default.
    """

    def __init__(
        self,
        filename: str | os.PathLike,
        parse_dos: bool = True,
        parse_eigen: bool = True,
        parse_projected_eigen: bool = False,
        # accepted for API compatibility, not used
        ionic_step_skip=None,
        ionic_step_offset: int = 0,
        parse_potcar_file=True,
        occu_tol: float = 1e-8,
        separate_spins: bool = False,
        exception_on_bad_xml: bool = True,
    ):
        self._raw = _RustVasprun(
            str(filename),
            parse_projected=parse_projected_eigen,
        )
        self._filename = Path(filename)

    # ---- basic metadata ----------------------------------------------------

    @property
    def incar(self) -> dict:
        return self._raw.incar

    @property
    def parameters(self) -> dict:
        """Alias for incar (pymatgen compatibility)."""
        return self._raw.incar

    @property
    def efermi(self) -> float | None:
        return self._raw.efermi

    @property
    def is_spin(self) -> bool:
        shape = self._raw.eigenvalue_shape
        return shape is not None and shape[0] == 2

    # ---- structures --------------------------------------------------------

    @property
    def final_structure(self) -> "Structure":
        return _pmg_structure(self._raw.final_structure)

    @property
    def initial_structure(self) -> "Structure":
        return _pmg_structure(self._raw.initial_structure)

    @property
    def structures(self) -> list["Structure"]:
        """All ionic-step structures, plus the final structure."""
        structs = [_pmg_structure(s["structure"]) for s in self._raw.ionic_steps]
        # append final if it differs from the last ionic step
        structs.append(self.final_structure)
        return structs

    @property
    def atomic_symbols(self) -> list[str]:
        return self._raw.atoms

    # ---- convergence -------------------------------------------------------

    @property
    def converged(self) -> bool:
        return self._raw.converged

    @property
    def converged_electronic(self) -> bool:
        return self._raw.converged

    @property
    def converged_ionic(self) -> bool:
        return self._raw.converged

    # ---- energy ------------------------------------------------------------

    @property
    def final_energy(self) -> float:
        steps = self._raw.ionic_steps
        if not steps:
            return float("nan")
        return steps[-1]["e_fr_energy"]

    @property
    def ionic_steps(self) -> list[dict]:
        return self._raw.ionic_steps

    # ---- k-points ----------------------------------------------------------

    @property
    def actual_kpoints(self) -> list[list[float]]:
        return self._raw.kpoints["kpointlist"].tolist()

    @property
    def actual_kpoints_weights(self) -> list[float]:
        return self._raw.kpoints["weights"].tolist()

    # ---- eigenvalues -------------------------------------------------------

    @property
    def eigenvalues(self) -> dict | None:
        """
        Dict mapping ``Spin`` → ``np.ndarray(shape=(nkpts, nbands, 2))``.
        Axis-2 is [energy, occupancy], matching pymatgen's format exactly.
        """
        raw = self._raw.eigenvalues
        if raw is None:
            return None
        from pymatgen.electronic_structure.core import Spin
        spins = _pmg_spin(raw.shape[0])
        # raw shape: (nspins, nkpts, nbands, 2) → per-spin: (nkpts, nbands, 2)
        return {spin: raw[i] for i, spin in enumerate(spins)}

    @property
    def projected_eigenvalues(self) -> dict | None:
        """
        Dict mapping ``Spin`` → ``np.ndarray(shape=(nkpts, nbands, nions, norbitals))``.
        Returns None if parse_projected_eigen was False.
        """
        proj = self._raw.projected
        if proj is None:
            return None
        from pymatgen.electronic_structure.core import Spin
        data = proj["data"]   # (nspins, nkpts, nbands, nions, norbitals)
        spins = _pmg_spin(data.shape[0])
        return {spin: data[i] for i, spin in enumerate(spins)}

    # ---- DOS ---------------------------------------------------------------

    @property
    def complete_dos(self) -> "CompleteDos":
        return self.get_complete_dos()

    @property
    def tdos(self) -> "Dos":
        return self._make_tdos()

    @property
    def idos(self) -> "Dos":
        from pymatgen.electronic_structure.dos import Dos
        from pymatgen.electronic_structure.core import Spin
        dos_raw = self._raw.dos
        if dos_raw is None:
            raise RuntimeError("No DOS data in this vasprun.xml")
        energies = dos_raw["total"]["energies"]
        idos_arr = dos_raw["total"]["integrated"]
        spins = _pmg_spin(idos_arr.shape[0])
        densities = {spin: idos_arr[i] for i, spin in enumerate(spins)}
        return Dos(self.efermi, energies, densities)

    def _make_tdos(self) -> "Dos":
        from pymatgen.electronic_structure.dos import Dos
        from pymatgen.electronic_structure.core import Spin
        dos_raw = self._raw.dos
        if dos_raw is None:
            raise RuntimeError("No DOS data in this vasprun.xml")
        energies = dos_raw["total"]["energies"]
        dens_arr = dos_raw["total"]["densities"]
        spins = _pmg_spin(dens_arr.shape[0])
        densities = {spin: dens_arr[i] for i, spin in enumerate(spins)}
        return Dos(self.efermi, energies, densities)

    def get_complete_dos(
        self,
        structure: "Structure | None" = None,
        integrated_dos: bool = False,
    ) -> "CompleteDos":
        from pymatgen.electronic_structure.dos import CompleteDos, Dos
        from pymatgen.electronic_structure.core import Spin, OrbitalType, Orbital
        dos_raw = self._raw.dos
        if dos_raw is None:
            raise RuntimeError("No DOS data in this vasprun.xml")

        struct = structure or self.final_structure
        tdos = self._make_tdos()

        pdos_dict: dict = {}
        if "partial" in dos_raw and dos_raw["partial"] is not None:
            partial = dos_raw["partial"]
            data = partial["data"]          # (nspins, nions, norbitals, nedos)
            orb_labels = partial["orbitals"]
            spins = _pmg_spin(data.shape[0])

            nions_in_dos = data.shape[1]
            for i, site in enumerate(struct):
                if i >= nions_in_dos:
                    break
                pdos_dict[site] = {}
                for oi, olabel in enumerate(orb_labels):
                    try:
                        orbital = Orbital[_VASP_ORBITAL_MAP.get(olabel, olabel)]
                    except KeyError:
                        continue
                    dens = {spin: data[si, i, oi] for si, spin in enumerate(spins)}
                    pdos_dict[site][orbital] = dens

        return CompleteDos(struct, tdos, pdos_dict)

    # ---- band structure ----------------------------------------------------

    def get_band_structure(
        self,
        kpoints_filename: str | os.PathLike | None = None,
        efermi: float | None = None,
        line_mode: bool = False,
        efermi_to_vbm: bool = False,
    ) -> "BandStructure | BandStructureSymmLine":
        from pymatgen.electronic_structure.bandstructure import (
            BandStructure,
            BandStructureSymmLine,
        )
        from pymatgen.electronic_structure.core import Spin
        from pymatgen.io.vasp.inputs import Kpoints

        eigs = self.eigenvalues
        if eigs is None:
            raise RuntimeError("No eigenvalue data in this vasprun.xml")

        efermi = efermi if efermi is not None else self.efermi
        lattice_rec = self.final_structure.lattice.reciprocal_lattice
        kpts_raw = self._raw.kpoints
        kpoints = kpts_raw["kpointlist"].tolist()

        # Build eigenvalues dict in pymatgen's expected format:
        # {Spin: np.ndarray(nkpts, nbands)}
        eigenvals = {spin: arr[..., 0] for spin, arr in eigs.items()}

        # Labels for high-symmetry points (line-mode calculations)
        labels_dict: dict[str, list[float]] = {}
        kpt_labels = kpts_raw.get("labels", [])
        if kpt_labels:
            for idx, label in kpt_labels:
                if label.strip():
                    labels_dict[label] = kpoints[idx]
            return BandStructureSymmLine(
                kpoints=kpoints,
                eigenvals=eigenvals,
                lattice=lattice_rec,
                efermi=efermi,
                labels_dict=labels_dict,
                structure=self.final_structure,
            )

        return BandStructure(
            kpoints=kpoints,
            eigenvals=eigenvals,
            lattice=lattice_rec,
            efermi=efermi,
            structure=self.final_structure,
        )

    # ---- dielectric --------------------------------------------------------

    @property
    def dielectric(self) -> tuple | None:
        """Returns (energies, real, imag) as lists, matching pymatgen's format."""
        raw = self._raw.dielectric
        if raw is None:
            return None
        return (
            raw["energies"].tolist(),
            raw["real"].tolist(),
            raw["imag"].tolist(),
        )

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Vasprunrs({self._filename.name!r}, "
            f"atoms={self.atomic_symbols}, "
            f"nkpts={len(self.actual_kpoints)}, "
            f"efermi={self.efermi:.4f})"
        )
