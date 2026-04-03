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

import hashlib
import itertools
import math
import os
import re
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
        ionic_step_skip=None,
        ionic_step_offset: int = 0,
        # accepted for API compatibility, ignored
        parse_potcar_file=True,
        occu_tol: float = 1e-8,
        separate_spins: bool = False,
        exception_on_bad_xml: bool = True,
    ):
        self._raw = _RustVasprun(
            str(filename),
            parse_dos=parse_dos,
            parse_eigen=parse_eigen,
            parse_projected=parse_projected_eigen,
            ionic_step_skip=ionic_step_skip,
            ionic_step_offset=ionic_step_offset,
        )
        self.filename = Path(filename)
        self._filename = self.filename  # backwards compat
        self.ionic_step_skip = ionic_step_skip
        self.ionic_step_offset = ionic_step_offset
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins
        self.exception_on_bad_xml = exception_on_bad_xml
        self.parse_dos = parse_dos
        self.parse_eigen = parse_eigen
        self.parse_projected_eigen = parse_projected_eigen
        self.parse_potcar_file = parse_potcar_file
        self.nionic_steps: int = len(self._raw.ionic_steps)

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

    # ---- magnetization -----------------------------------------------------

    @property
    def magnetization(self) -> list[list[float]] | None:
        """Per-atom magnetic moments from the final ionic step.

        Returns a list with one entry per atom.  Each entry is a list of
        floats: one value for collinear spin (ISPIN=2, total moment in μB),
        or three values for non-collinear calculations (mx, my, mz in μB).

        Returns ``None`` when the calculation is non-magnetic (magnetization
        block absent from vasprun.xml).
        """
        steps = self._raw.ionic_steps
        if not steps:
            return None
        return steps[-1].get("magnetization")

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

    @property
    def kpoints(self):
        """
        Returns a ``pymatgen.io.vasp.inputs.Kpoints`` object constructed from
        the parsed XML data, including high-symmetry labels for line-mode runs.
        Mirrors the ``Vasprun.kpoints`` attribute.
        """
        from pymatgen.io.vasp.inputs import Kpoints, KpointsSupportedModes

        kpts_raw = self._raw.kpoints
        kpointlist = kpts_raw["kpointlist"].tolist()
        weights    = kpts_raw["weights"].tolist()
        nkpts      = len(kpointlist)

        # Convert [(idx, label), ...] → per-kpoint label list (None if absent).
        kpt_labels_raw = kpts_raw.get("labels", [])
        if kpt_labels_raw:
            labels: list[str] | None = [""] * nkpts
            for idx, label in kpt_labels_raw:
                labels[idx] = label
        else:
            labels = None

        # vasprun.xml stores the actual expanded k-point list (one row per kpt),
        # so we always use Reciprocal style for the pymatgen object regardless of
        # how the grid was generated.  Gamma/Monkhorst styles expect a single
        # [nx, ny, nz] divisions entry and reject explicit lists.
        scheme = kpts_raw.get("scheme", "")
        style = (
            KpointsSupportedModes.Line_mode
            if scheme in ("Line-mode", "line", "L")
            else KpointsSupportedModes.Reciprocal
        )

        return Kpoints(
            comment="Generated by vasprunrs",
            num_kpts=nkpts,
            style=style,
            kpts=kpointlist,
            kpts_weights=weights,
            labels=labels,
        )

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

    # ---- functional / DFT+U -----------------------------------------------

    @property
    def run_type(self) -> str:
        """Functional type: GGA, PBE, SCAN, HSE06, HF, etc., with optional +U / +vdW suffix.

        Derived from INCAR tags only (not POTCAR). When no functional tag is
        explicitly set the calculation is assumed to be GGA (PBE via POTCAR).
        """
        GGA_TYPES = {
            "RE": "revPBE", "PE": "PBE", "PS": "PBEsol",
            "RP": "revPBE+Padé", "AM": "AM05", "OR": "optPBE",
            "BO": "optB88", "MK": "optB86b", "--": "GGA",
        }
        METAGGA_TYPES = {
            "TPSS": "TPSS", "RTPSS": "revTPSS", "M06L": "M06-L",
            "MBJ": "modified Becke-Johnson", "SCAN": "SCAN",
            "R2SCAN": "R2SCAN", "RSCAN": "RSCAN",
            "MS0": "MadeSimple0", "MS1": "MadeSimple1", "MS2": "MadeSimple2",
        }
        IVDW_TYPES = {
            0: "no-correction", 1: "DFT-D2", 10: "DFT-D2",
            11: "DFT-D3", 12: "DFT-D3-BJ", 2: "TS", 20: "TS",
            21: "TS-H", 202: "MBD", 4: "dDsC",
        }

        p = self._raw.incar
        if p.get("LHFCALC"):
            aexx    = p.get("AEXX",    0.25)
            hfscreen = p.get("HFSCREEN", 0.0)
            if math.isclose(aexx, 1.0):
                run_type = "HF"
            elif math.isclose(hfscreen, 0.2):
                run_type = "HSE06"
            elif math.isclose(hfscreen, 0.3):
                run_type = "HSE03"
            elif math.isclose(aexx, 0.2):
                run_type = "B3LYP"
            else:
                run_type = "PBE0 or other Hybrid Functional"
        elif p.get("METAGGA") and str(p.get("METAGGA", "")).strip().upper() not in {"--", "NONE", ""}:
            tag = str(p.get("METAGGA", "")).strip().upper()
            run_type = METAGGA_TYPES.get(tag, tag)
        elif p.get("GGA"):
            tag = str(p.get("GGA", "")).strip().upper()
            run_type = GGA_TYPES.get(tag, tag)
        else:
            run_type = "GGA"

        if self.is_hubbard:
            run_type += "+U"
        if p.get("LUSE_VDW"):
            run_type += "+rVV10"
        elif p.get("IVDW") in IVDW_TYPES:
            run_type += f"+vdW-{IVDW_TYPES[p['IVDW']]}"
        elif p.get("IVDW"):
            run_type += "+vdW-unknown"
        return run_type

    @property
    def hubbards(self) -> dict[str, float]:
        """Hubbard U−J values per species for DFT+U runs, otherwise empty dict."""
        if not self._raw.incar.get("LDAU"):
            return {}
        species = [at["element"] for at in self._raw.atom_types]
        us = self._raw.incar.get("LDAUU", [])
        js = self._raw.incar.get("LDAUJ", [])
        if not isinstance(us, list):
            us = [us]
        if not isinstance(js, list):
            js = [js]
        if len(js) != len(us):
            js = [0.0] * len(us)
        if len(us) == len(species):
            return {species[i]: us[i] - js[i] for i in range(len(species))}
        return {}

    @property
    def is_hubbard(self) -> bool:
        """Whether this is a DFT+U calculation."""
        return bool(self.hubbards) and sum(self.hubbards.values()) > 1e-8

    @property
    def potcar_symbols(self) -> list[str]:
        """Best-effort POTCAR symbol list derived from atom types.

        Not parsed from an actual POTCAR file — returns ``"PAW_PBE <element>"``
        for each species.  Accepted for API compatibility; use with caution.
        """
        return [f"PAW_PBE {at['element']}" for at in self._raw.atom_types]

    # ---- ionic step count --------------------------------------------------

    @property
    def md_n_steps(self) -> int:
        """Number of ionic steps (alias for ``nionic_steps``)."""
        return self.nionic_steps

    # ---- band properties ---------------------------------------------------

    @property
    def eigenvalue_band_properties(self):
        """Band gap, CBM, VBM, and whether the gap is direct.

        Returns ``(band_gap, cbm, vbm, is_direct)``.  With ``separate_spins=True``
        each element is a 2-tuple ``(spin_up, spin_down)``.

        Returns ``(0.0, 0.0, 0.0, False)`` when eigenvalues are absent.
        """
        eigs = self.eigenvalues
        if eigs is None:
            return (0.0, 0.0, 0.0, False)

        vbm = -float("inf")
        cbm =  float("inf")
        vbm_kpoint = cbm_kpoint = None

        vbm_spins, cbm_spins = [], []
        vbm_kpts,  cbm_kpts  = [], []

        for eigenvalue in eigs.values():
            if self.separate_spins:
                vbm = -float("inf")
                cbm =  float("inf")
            for kpoint, val in enumerate(eigenvalue):
                for eigenval, occ in val:
                    if occ > self.occu_tol and eigenval > vbm:
                        vbm = eigenval;  vbm_kpoint = kpoint
                    elif occ <= self.occu_tol and eigenval < cbm:
                        cbm = eigenval;  cbm_kpoint = kpoint
            if self.separate_spins:
                vbm_spins.append(vbm);  vbm_kpts.append(vbm_kpoint)
                cbm_spins.append(cbm);  cbm_kpts.append(cbm_kpoint)

        if self.separate_spins:
            return (
                (max(cbm_spins[0] - vbm_spins[0], 0), max(cbm_spins[1] - vbm_spins[1], 0)),
                (cbm_spins[0], cbm_spins[1]),
                (vbm_spins[0], vbm_spins[1]),
                (vbm_kpts[0] == cbm_kpts[0], vbm_kpts[1] == cbm_kpts[1]),
            )
        return max(cbm - vbm, 0), cbm, vbm, vbm_kpoint == cbm_kpoint

    def calculate_efermi(self, tol: float = 0.001) -> float:
        """Recalculate the Fermi level from eigenvalue occupancies.

        Corrects cases where VASP places the Fermi level inside a band due to
        tetrahedron integration artefacts.
        """
        eigs = self.eigenvalues
        if eigs is None:
            raise ValueError("eigenvalues are not available")
        if self.efermi is None:
            raise ValueError("efermi is not available")

        all_eigs = np.concatenate([arr[:, :, 0].T for arr in eigs.values()])

        def crosses_band(fermi: float) -> bool:
            below = np.any(all_eigs < fermi, axis=1)
            above = np.any(all_eigs > fermi, axis=1)
            return bool(np.any(below & above))

        def get_vbm_cbm(fermi: float):
            return np.max(all_eigs[all_eigs < fermi]), np.min(all_eigs[all_eigs > fermi])

        if not crosses_band(self.efermi):
            return self.efermi
        if not crosses_band(self.efermi + tol):
            vbm, cbm = get_vbm_cbm(self.efermi + tol)
            return float((cbm + vbm) / 2)
        if not crosses_band(self.efermi - tol):
            vbm, cbm = get_vbm_cbm(self.efermi - tol)
            return float((cbm + vbm) / 2)
        return float(self.efermi)

    # ---- dielectric / optical ----------------------------------------------

    @property
    def epsilon_static(self) -> list:
        """Static dielectric tensor from DFPT (LEPSILON=True). Empty if not computed."""
        return self.ionic_steps[-1].get("epsilon", []) if self.ionic_steps else []

    @property
    def epsilon_ionic(self) -> list:
        """Ionic contribution to the static dielectric tensor (IBRION=5–8). Empty if not computed."""
        return self.ionic_steps[-1].get("epsilon_ion", []) if self.ionic_steps else []

    @property
    def epsilon_static_wolfe(self) -> list:
        """Wolfe-method static dielectric tensor. Empty if not computed."""
        return self.ionic_steps[-1].get("epsilon_wolfe", []) if self.ionic_steps else []

    @property
    def optical_absorption_coeff(self) -> list[float] | None:
        """Optical absorption coefficient (cm⁻¹) derived from the dielectric function.

        Computed from the isotropic average of the diagonal components.
        Returns ``None`` when no dielectric data is available.
        """
        raw = self._raw.dielectric
        if raw is None:
            return None
        energies = raw["energies"].tolist()
        real_avg = [float(np.mean(raw["real"][i, :3])) for i in range(len(energies))]
        imag_avg = [float(np.mean(raw["imag"][i, :3])) for i in range(len(energies))]
        hc = 1.23984e-4  # eV·cm

        def _coeff(freq: float, real: float, imag: float) -> float:
            return 2 * math.pi * math.sqrt(math.sqrt(real**2 + imag**2) - real) * math.sqrt(2) / hc * freq

        return list(itertools.starmap(_coeff, zip(energies, real_avg, imag_avg)))

    # ---- normalized DOS ----------------------------------------------------

    @property
    def complete_dos_normalized(self):
        """CompleteDos normalized by unit-cell volume (states/eV/Å³)."""
        from pymatgen.electronic_structure.dos import CompleteDos
        cdos = self.complete_dos
        vol = self.final_structure.volume
        tdos_norm = type(cdos.densities)(  # same Spin-keyed dict
            {spin: dens / vol for spin, dens in cdos.densities.items()}
        )
        from pymatgen.electronic_structure.dos import Dos
        tdos = Dos(cdos.efermi, cdos.energies, tdos_norm)
        pdos_norm = {
            site: {orb: {spin: d / vol for spin, d in dens.items()}
                   for orb, dens in orb_dos.items()}
            for site, orb_dos in cdos.pdos.items()
        }
        return CompleteDos(self.final_structure, tdos, pdos_norm)

    # ---- projected magnetization -------------------------------------------

    @property
    def projected_magnetization(self) -> np.ndarray | None:
        """Projected magnetization array, shape ``(nkpts, nbands, nions, norbitals)``.

        Only available for non-collinear calculations with ``parse_projected_eigen=True``.
        Returns ``None`` otherwise.
        """
        proj = self._raw.projected
        if proj is None or not self.is_spin:
            return None
        data = proj["data"]   # (nspins, nkpts, nbands, nions, norbitals)
        if data.shape[0] < 2:
            return None
        return data[0] - data[1]

    # ---- computed entry ----------------------------------------------------

    def get_computed_entry(
        self,
        inc_structure: bool = True,
        parameters: list[str] | None = None,
        data: list[str] | None = None,
        entry_id: str | None = None,
    ):
        """Return a ``ComputedStructureEntry`` (or ``ComputedEntry``) for this calculation.

        Parameters
        ----------
        inc_structure:
            If True (default), return a ``ComputedStructureEntry`` which embeds
            the final structure.
        parameters:
            Additional ``Vasprunrs`` property names to store in the entry's
            parameters dict.  Defaults to ``{"is_hubbard", "hubbards",
            "potcar_symbols", "run_type"}``.
        data:
            ``Vasprunrs`` property names to store in the entry's data dict.
        entry_id:
            Explicit entry ID.  Auto-generated from file name + structure hash
            when omitted.
        """
        from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

        if entry_id is None:
            struct_hash = hashlib.md5(str(self.final_structure).encode()).hexdigest()[:8]  # noqa: S324
            entry_id = f"vasprun-{self.filename.stem}-{struct_hash}"

        param_names = {"is_hubbard", "hubbards", "potcar_symbols", "run_type"}
        if parameters:
            param_names.update(parameters)
        params = {k: getattr(self, k) for k in param_names}
        _data = {k: getattr(self, k) for k in (data or [])}

        if inc_structure:
            return ComputedStructureEntry(
                self.final_structure, self.final_energy,
                parameters=params, data=_data, entry_id=entry_id,
            )
        return ComputedEntry(
            self.final_structure.composition, self.final_energy,
            parameters=params, data=_data, entry_id=entry_id,
        )

    # ---- trajectory --------------------------------------------------------

    def get_trajectory(self):
        """Return a pymatgen ``Trajectory`` built from the ionic steps."""
        from pymatgen.core.trajectory import Trajectory
        structs = []
        for step in self.ionic_steps:
            s = _pmg_structure(step["structure"])
            s.add_site_property("forces", step["forces"])
            structs.append(s)
        return Trajectory.from_structures(structs, constant_lattice=False)

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Vasprunrs({self._filename.name!r}, "
            f"atoms={self.atomic_symbols}, "
            f"nkpts={len(self.actual_kpoints)}, "
            f"efermi={self.efermi:.4f})"
        )
