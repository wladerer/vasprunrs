"""Type stubs for the Rust-compiled vasprunrs extension module."""

from __future__ import annotations

import numpy as np
from typing import Optional

class Vasprun:
    """Fast VASP vasprun.xml parser backed by a Rust streaming XML parser.

    Parameters
    ----------
    path:
        Path to the vasprun.xml file.
    parse_dos:
        Parse density of states (default True).
    parse_eigen:
        Parse eigenvalues (default True).
    parse_projected:
        Parse projected eigenvalues, i.e. LORBIT >= 10 (default False).
    ionic_step_skip:
        If set, parse every Nth ionic step only.
    ionic_step_offset:
        Skip the first N ionic steps.
    """

    def __init__(
        self,
        path: str,
        parse_dos: bool = True,
        parse_eigen: bool = True,
        parse_projected: bool = False,
        ionic_step_skip: Optional[int] = None,
        ionic_step_offset: int = 0,
    ) -> None: ...

    # --- generator -----------------------------------------------------------

    @property
    def program(self) -> str:
        """VASP program name, e.g. ``"vasp"``."""
        ...

    @property
    def version(self) -> str:
        """VASP version string, e.g. ``"6.4.2"``."""
        ...

    # --- INCAR ---------------------------------------------------------------

    @property
    def incar(self) -> dict[str, int | float | bool | str | list[float]]:
        """INCAR parameters with native Python types (int, float, bool, str, list)."""
        ...

    # --- atominfo ------------------------------------------------------------

    @property
    def atoms(self) -> list[str]:
        """Element symbol for each atomic site, e.g. ``["Si", "Si"]``."""
        ...

    @property
    def atom_types(self) -> list[dict[str, object]]:
        """Per-species metadata.

        Each entry is a dict with keys:
        ``element`` (str), ``count`` (int), ``mass`` (float),
        ``valence`` (float), ``pseudopotential`` (str).
        """
        ...

    # --- k-points ------------------------------------------------------------

    @property
    def kpoints(self) -> dict[str, object]:
        """K-point information.

        Always present keys:
        - ``"kpointlist"``: ``np.ndarray`` of shape ``(nkpts, 3)`` in reciprocal coordinates
        - ``"weights"``: ``np.ndarray`` of shape ``(nkpts,)`` normalized to 1.0

        Optional keys (when present in the file):
        - ``"scheme"``: str — generation scheme, e.g. ``"Gamma"`` or ``"Monkhorst-Pack"``
        - ``"divisions"``: list[int] — k-mesh divisions
        - ``"labels"``: list[tuple[int, str]] — high-symmetry labels as ``(index, name)``
        """
        ...

    # --- structures ----------------------------------------------------------

    @property
    def initial_structure(self) -> dict[str, object]:
        """Initial crystal structure.

        Keys: ``"lattice"`` (list[list[float]], 3x3 in Å),
        ``"volume"`` (float), ``"positions"`` (list[list[float]], fractional),
        ``"species"`` (list[str]).
        """
        ...

    @property
    def final_structure(self) -> dict[str, object]:
        """Final crystal structure (same schema as ``initial_structure``)."""
        ...

    # --- ionic steps ---------------------------------------------------------

    @property
    def ionic_steps(self) -> list[dict[str, object]]:
        """List of ionic steps.

        Each dict has keys:
        - ``"structure"``: same schema as ``initial_structure``
        - ``"e_fr_energy"``: float — free energy (eV)
        - ``"e_wo_entrp"``: float — energy without entropy (eV)
        - ``"e_0_energy"``: float — energy at sigma→0 (eV)
        - ``"forces"``: list[list[float]] — shape (natoms, 3), eV/Å
        - ``"stress"``: list[list[float]] — shape (3, 3), kbar
        - ``"magnetization"`` (optional): list[list[float]] — per-atom magnetic
          moments; inner list has 1 element for collinear (ISPIN=2) or 3 for
          non-collinear (mx, my, mz). Absent for non-magnetic calculations.
        """
        ...

    # --- eigenvalues ---------------------------------------------------------

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """Eigenvalues array of shape ``(nspins, nkpts, nbands, 2)``.

        Axis 3: ``[energy (eV), occupancy]``.
        Returns ``None`` when ``parse_eigen=False``.
        """
        ...

    @property
    def eigenvalue_shape(self) -> Optional[tuple[int, int, int]]:
        """``(nspins, nkpts, nbands)`` or ``None`` when not parsed."""
        ...

    # --- projected eigenvalues -----------------------------------------------

    @property
    def projected(self) -> Optional[dict[str, object]]:
        """Projected eigenvalues (LORBIT >= 10).

        Returns a dict with:
        - ``"data"``: ``np.ndarray`` of shape ``(nspins, nkpts, nbands, nions, norbitals)``
        - ``"orbitals"``: list[str] — orbital labels, e.g. ``["s", "py", "pz", "px", ...]``

        Returns ``None`` when ``parse_projected=False`` or not present.
        """
        ...

    # --- Fermi level ---------------------------------------------------------

    @property
    def efermi(self) -> Optional[float]:
        """Fermi level in eV, or ``None`` if not found in the file."""
        ...

    # --- DOS -----------------------------------------------------------------

    @property
    def dos(self) -> Optional[dict[str, object]]:
        """Density of states data.

        Returns a dict with:
        - ``"efermi"``: float
        - ``"total"``: dict with ``"energies"`` (ndarray), ``"densities"`` (ndarray, shape (nspins, nedos)), ``"integrated"`` (ndarray)
        - ``"partial"`` (optional): dict with ``"data"`` (ndarray, shape (nspins, nions, norbitals, nedos)) and ``"orbitals"`` (list[str])

        Returns ``None`` when ``parse_dos=False`` or DOS block absent.
        """
        ...

    # --- dielectric ----------------------------------------------------------

    @property
    def dielectric(self) -> Optional[dict[str, object]]:
        """Frequency-dependent dielectric function.

        Returns a dict with:
        - ``"energies"``: ndarray of shape (nfreq,)
        - ``"real"``: ndarray of shape (nfreq, 6) — xx yy zz xy yz zx
        - ``"imag"``: ndarray of shape (nfreq, 6)

        Returns ``None`` when not present in the file.
        """
        ...

    # --- convenience ---------------------------------------------------------

    @property
    def converged(self) -> bool:
        """``True`` if the last ionic step has a non-zero free energy."""
        ...

    def __repr__(self) -> str: ...
