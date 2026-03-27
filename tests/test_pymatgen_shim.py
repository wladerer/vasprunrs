"""
Pymatgen-shim parity tests for vasprunrs.

Each test runs the same file through both pymatgen's Vasprun and Vasprunrs and
asserts that the outputs agree within tolerance.  Numerical reference values are
derived from pymatgen and baked in as regression anchors so that the suite can
run without pymatgen installed (by skipping the diff-tests and keeping the value
tests).

Fixtures
--------
si_rpa       tests/si_rpa.xml         – Si, 1-step, total DOS only, dielectric
mos2_soc     tests/mos2_soc.xml       – MoS2 SOC, Wannier-interpolated dos block
n_gw         tests/n_gw_large.xml     – N GW post-processing, spin-polarised, no ionic steps
v4c3_geo     research mxene           – V4C3 geo-opt, 27 ionic steps, partial DOS (spd)
mxene_h_soc  research mxene+H (SOC)   – non-collinear, 4-component partial DOS

Known feature gaps (not tested here, tracked separately)
---------------------------------------------------------
- projected_eigenvalues: Rust parses them but Python binding does not yet expose
  the attribute; Vasprunrs.projected_eigenvalues always returns None.
- kpoints.labels: line-mode k-point labels are parsed in Rust but not included
  in the Python binding's kpoints dict, so get_band_structure() always returns
  a plain BandStructure rather than BandStructureSymmLine.
"""
import os
import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import vasprunrs
from vasprunrs.pymatgen import Vasprunrs

try:
    from pymatgen.io.vasp import Vasprun as PmgVasprun
    from pymatgen.electronic_structure.core import Spin
    HAS_PMG = True
except ImportError:
    HAS_PMG = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(__file__))

SI_RPA      = os.path.join(REPO, "tests/si_rpa.xml")
MOS2_SOC    = os.path.join(REPO, "tests/mos2_soc.xml")
N_GW        = os.path.join(REPO, "tests/n_gw_large.xml")
V4C3_GEO    = "/home/wladerer/research/mxenes/geo_opt/medium/vasprun.xml"
MXENE_H_SOC = "/home/wladerer/research/mxenes/soc_adsorption/capped/H/vasprun.xml"

needs_pmg     = pytest.mark.skipif(not HAS_PMG, reason="pymatgen not installed")
needs_v4c3    = pytest.mark.skipif(not os.path.exists(V4C3_GEO),    reason="V4C3 research file absent")
needs_mxene_h = pytest.mark.skipif(not os.path.exists(MXENE_H_SOC), reason="mxene-H research file absent")

TOL = 1e-4   # eV / Å tolerance for scalar comparisons
DOS_TOL = 1e-5  # max per-point difference in DOS densities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pmg(path):
    return PmgVasprun(path, parse_potcar_file=False)

def vrs(path):
    return Vasprunrs(path, parse_potcar_file=False)


# ===========================================================================
# Si RPA  (tests/si_rpa.xml)
# ===========================================================================
class TestSiRpa:
    def test_efermi(self):
        assert abs(vrs(SI_RPA).efermi - 6.326521) < TOL

    def test_structure_nions(self):
        assert len(vrs(SI_RPA).final_structure) == 2

    def test_structure_lattice(self):
        lat = vrs(SI_RPA).final_structure.lattice
        assert abs(lat.a - 3.80473) < TOL
        assert abs(lat.c - 3.80473) < TOL

    def test_species(self):
        syms = [s.symbol for s in vrs(SI_RPA).final_structure.species]
        assert syms == ["Si", "Si"]

    def test_converged(self):
        assert vasprunrs.Vasprun(SI_RPA).converged is True

    def test_ionic_steps(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        assert len(raw.ionic_steps) == 1
        assert abs(raw.ionic_steps[0]["e_fr_energy"] - (-11.690705)) < TOL

    def test_dos_nedos(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        assert len(raw.dos["total"]["energies"]) == 301

    def test_dos_energy_range(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        e = raw.dos["total"]["energies"]
        assert abs(e[0]  - (-9.3961)) < 1e-3
        assert abs(e[-1] -  64.0168)  < 1e-3

    def test_dielectric_present(self):
        assert vasprunrs.Vasprun(SI_RPA).dielectric is not None

    def test_kpoints(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        assert raw.kpoints["kpointlist"].shape[0] == 8

    @needs_pmg
    def test_parity_efermi(self):
        assert abs(vrs(SI_RPA).efermi - pmg(SI_RPA).efermi) < TOL

    @needs_pmg
    def test_parity_structure(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert len(v.final_structure) == len(p.final_structure)
        assert abs(v.final_structure.lattice.a - p.final_structure.lattice.a) < TOL

    @needs_pmg
    def test_parity_tdos(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        vd = v.tdos.densities[Spin.up]
        pd = p.tdos.densities[Spin.up]
        assert np.abs(vd - pd).max() < DOS_TOL


# ===========================================================================
# MoS2 SOC  (tests/mos2_soc.xml)
# Regression: Wannier-interpolated <dos> block must not steal the efermi.
# ===========================================================================
class TestMos2Soc:
    def test_efermi(self):
        # 3.893022, NOT the Wannier value 8.862877
        assert abs(vrs(MOS2_SOC).efermi - 3.893022) < TOL

    def test_structure(self):
        s = vrs(MOS2_SOC).final_structure
        assert len(s) == 3
        syms = sorted(sp.symbol for sp in s.species)
        assert syms == ["Mo", "S", "S"]
        assert abs(s.lattice.a - 3.18648) < TOL
        assert abs(s.lattice.c - 8.00000) < TOL

    def test_final_energy(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        assert abs(raw.ionic_steps[-1]["e_fr_energy"] - (-21.871868)) < TOL

    def test_kpoints(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        assert raw.kpoints["kpointlist"].shape[0] == 29

    def test_element_dos(self):
        cdos = vrs(MOS2_SOC).complete_dos
        el_dos = cdos.get_element_dos()
        assert set(str(e) for e in el_dos) == {"Mo", "S"}
        mo = el_dos[[e for e in el_dos if str(e) == "Mo"][0]].densities[Spin.up]
        s  = el_dos[[e for e in el_dos if str(e) == "S"][0]].densities[Spin.up]
        assert abs(mo.sum() - 68.8793) < 1.0
        assert abs(s.sum()  - 37.0556) < 1.0

    @needs_pmg
    def test_parity_efermi(self):
        assert abs(vrs(MOS2_SOC).efermi - pmg(MOS2_SOC).efermi) < TOL

    @needs_pmg
    def test_parity_element_dos(self):
        v, p = vrs(MOS2_SOC), pmg(MOS2_SOC)
        vel, pel = v.complete_dos.get_element_dos(), p.complete_dos.get_element_dos()
        for el in pel:
            vd = vel[el].densities[Spin.up]
            pd = pel[el].densities[Spin.up]
            assert np.abs(vd - pd).max() < DOS_TOL, f"{el} mismatch"


# ===========================================================================
# N GW  (tests/n_gw_large.xml)
# Regression: GW post-processing has no ionic steps and is spin-polarised.
# ===========================================================================
class TestNGw:
    def test_efermi(self):
        assert abs(vrs(N_GW).efermi - (-6.553045)) < TOL

    def test_structure(self):
        s = vrs(N_GW).final_structure
        assert len(s) == 1
        assert s[0].specie.symbol == "N"

    def test_no_ionic_steps(self):
        assert len(vasprunrs.Vasprun(N_GW).ionic_steps) == 0

    def test_not_converged(self):
        # GW calculations don't have a relaxation to converge
        assert vasprunrs.Vasprun(N_GW).converged is False

    def test_spin_polarised_eigenvalues(self):
        raw = vasprunrs.Vasprun(N_GW)
        nspins, _, _ = raw.eigenvalue_shape
        assert nspins == 2

    def test_kpoints(self):
        raw = vasprunrs.Vasprun(N_GW)
        assert raw.kpoints["kpointlist"].shape[0] == 4

    @needs_pmg
    def test_parity_efermi(self):
        assert abs(vrs(N_GW).efermi - pmg(N_GW).efermi) < TOL

    @needs_pmg
    def test_parity_structure(self):
        v, p = vrs(N_GW), pmg(N_GW)
        assert len(v.final_structure) == len(p.final_structure)


# ===========================================================================
# V4C3 geo-opt  (research file)
# Covers: multi-step relaxation, partial DOS (spd, 1 spin), dx2-y2 orbital.
# ===========================================================================
@needs_v4c3
class TestV4C3Geo:
    def test_efermi(self):
        assert abs(vrs(V4C3_GEO).efermi - (-0.216826)) < TOL

    def test_structure(self):
        s = vrs(V4C3_GEO).final_structure
        assert len(s) == 7
        syms = sorted(sp.symbol for sp in s.species)
        assert syms == ["C", "C", "C", "V", "V", "V", "V"]
        assert abs(s.lattice.a -  2.92337) < TOL
        assert abs(s.lattice.c - 28.18367) < TOL

    def test_ionic_steps(self):
        raw = vasprunrs.Vasprun(V4C3_GEO)
        assert len(raw.ionic_steps) == 27
        assert abs(raw.ionic_steps[-1]["e_fr_energy"] - (-55.688552)) < TOL

    def test_converged(self):
        assert vasprunrs.Vasprun(V4C3_GEO).converged is True

    def test_partial_dos_shape(self):
        raw = vasprunrs.Vasprun(V4C3_GEO)
        # (nspins=1, nions=7, norbitals=9, nedos=301)
        assert raw.dos["partial"]["data"].shape == (1, 7, 9, 301)

    def test_element_dos_dx2(self):
        """dx2-y2 orbital must not be silently dropped (regression)."""
        cdos = vrs(V4C3_GEO).complete_dos
        el_dos = cdos.get_element_dos()
        v_dos = el_dos[[e for e in el_dos if str(e) == "V"][0]].densities[Spin.up]
        assert abs(v_dos.sum() - 630.4972) < 1.0

    def test_element_dos_carbon(self):
        cdos = vrs(V4C3_GEO).complete_dos
        el_dos = cdos.get_element_dos()
        c_dos = el_dos[[e for e in el_dos if str(e) == "C"][0]].densities[Spin.up]
        assert abs(c_dos.sum() - 143.8505) < 1.0

    @needs_pmg
    def test_parity_efermi(self):
        assert abs(vrs(V4C3_GEO).efermi - pmg(V4C3_GEO).efermi) < TOL

    @needs_pmg
    def test_parity_element_dos(self):
        v, p = vrs(V4C3_GEO), pmg(V4C3_GEO)
        vel, pel = v.complete_dos.get_element_dos(), p.complete_dos.get_element_dos()
        for el in pel:
            vd = vel[el].densities[Spin.up]
            pd = pel[el].densities[Spin.up]
            assert np.abs(vd - pd).max() < DOS_TOL, f"{el} mismatch"

    @needs_pmg
    def test_parity_structure(self):
        v, p = vrs(V4C3_GEO), pmg(V4C3_GEO)
        assert len(v.final_structure) == len(p.final_structure)
        assert abs(v.final_structure.lattice.a - p.final_structure.lattice.a) < TOL


# ===========================================================================
# MXene + H SOC  (research file)
# Covers: non-collinear (4-component) partial DOS, multi-element slab.
# ===========================================================================
@needs_mxene_h
class TestMxeneHSoc:
    def test_efermi(self):
        assert abs(vrs(MXENE_H_SOC).efermi - 0.019154) < TOL

    def test_structure(self):
        s = vrs(MXENE_H_SOC).final_structure
        assert len(s) == 10
        syms = sorted(set(sp.symbol for sp in s.species))
        assert syms == ["C", "H", "O", "V"]

    def test_converged(self):
        assert vasprunrs.Vasprun(MXENE_H_SOC).converged is True

    def test_final_energy(self):
        raw = vasprunrs.Vasprun(MXENE_H_SOC)
        assert abs(raw.ionic_steps[-1]["e_fr_energy"] - (-74.644549)) < TOL

    def test_kpoints(self):
        raw = vasprunrs.Vasprun(MXENE_H_SOC)
        assert raw.kpoints["kpointlist"].shape[0] == 324

    def test_partial_dos_shape(self):
        raw = vasprunrs.Vasprun(MXENE_H_SOC)
        # (nspins=4 non-collinear, nions=10, norbitals=9, nedos=301)
        assert raw.dos["partial"]["data"].shape == (4, 10, 9, 301)

    def test_element_dos_all_present(self):
        cdos = vrs(MXENE_H_SOC).complete_dos
        el_syms = {str(e) for e in cdos.get_element_dos()}
        assert el_syms == {"V", "H", "C", "O"}

    def test_element_dos_values(self):
        cdos = vrs(MXENE_H_SOC).complete_dos
        el_dos = cdos.get_element_dos()
        expected = {"V": 480.3891, "H": 12.7499, "C": 99.9611, "O": 146.8939}
        for el_obj, exp_sum in expected.items():
            match = [e for e in el_dos if str(e) == el_obj]
            assert match, f"Element {el_obj} missing from complete_dos"
            d = el_dos[match[0]].densities[Spin.up]
            assert abs(d.sum() - exp_sum) < 1.0, f"{el_obj}: got {d.sum():.4f}, expected {exp_sum}"

    @needs_pmg
    def test_parity_efermi(self):
        assert abs(vrs(MXENE_H_SOC).efermi - pmg(MXENE_H_SOC).efermi) < TOL

    @needs_pmg
    def test_parity_element_dos(self):
        v, p = vrs(MXENE_H_SOC), pmg(MXENE_H_SOC)
        vel, pel = v.complete_dos.get_element_dos(), p.complete_dos.get_element_dos()
        for el in pel:
            vd = vel[el].densities[Spin.up]
            pd = pel[el].densities[Spin.up]
            assert np.abs(vd - pd).max() < DOS_TOL, f"{el} mismatch"

    @needs_pmg
    def test_parity_structure(self):
        v, p = vrs(MXENE_H_SOC), pmg(MXENE_H_SOC)
        assert len(v.final_structure) == len(p.final_structure)
        assert abs(v.final_structure.lattice.a - p.final_structure.lattice.a) < TOL


# ===========================================================================
# Cross-fixture: eigenvalues, dielectric, forces, INCAR, kpoints
# ===========================================================================
class TestEigenvaluesParity:
    """Eigenvalue array matches pymatgen exactly for all bundled fixtures."""

    @needs_pmg
    def test_si_rpa(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        from pymatgen.electronic_structure.core import Spin
        assert np.abs(v.eigenvalues[Spin.up] - p.eigenvalues[Spin.up]).max() == 0.0

    @needs_pmg
    def test_mos2_soc(self):
        v, p = vrs(MOS2_SOC), pmg(MOS2_SOC)
        from pymatgen.electronic_structure.core import Spin
        assert np.abs(v.eigenvalues[Spin.up] - p.eigenvalues[Spin.up]).max() == 0.0

    @needs_pmg
    def test_n_gw_spin_polarised(self):
        v, p = vrs(N_GW), pmg(N_GW)
        from pymatgen.electronic_structure.core import Spin
        for spin in (Spin.up, Spin.down):
            assert np.abs(v.eigenvalues[spin] - p.eigenvalues[spin]).max() == 0.0


class TestDielectricParity:
    """Dielectric function (energies, real, imag) matches pymatgen exactly."""

    @needs_pmg
    def test_si_rpa_energies(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert np.allclose(v.dielectric[0], p.dielectric[0])

    @needs_pmg
    def test_si_rpa_real(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert np.allclose(v.dielectric[1], p.dielectric[1])

    @needs_pmg
    def test_si_rpa_imag(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert np.allclose(v.dielectric[2], p.dielectric[2])

    def test_shape(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        diel = raw.dielectric
        assert diel["real"].shape == (1000, 6)
        assert diel["imag"].shape == (1000, 6)
        assert len(diel["energies"]) == 1000


class TestForcesAndStress:
    """Forces and stress are parsed and have the right shape/values."""

    def test_si_rpa_forces_shape(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        forces = raw.ionic_steps[0]["forces"]
        assert len(forces) == 2        # 2 Si atoms
        assert len(forces[0]) == 3     # 3 Cartesian components

    def test_si_rpa_stress_shape(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        stress = raw.ionic_steps[0]["stress"]
        assert len(stress) == 3 and len(stress[0]) == 3

    def test_si_rpa_stress_isotropic(self):
        # cubic Si at equilibrium — stress should be diagonal and nearly equal
        stress = vasprunrs.Vasprun(SI_RPA).ionic_steps[0]["stress"]
        assert abs(stress[0][0] - stress[1][1]) < 1e-4
        assert abs(stress[0][0] - stress[2][2]) < 1e-4

    @needs_v4c3
    def test_v4c3_forces_shape(self):
        raw = vasprunrs.Vasprun(V4C3_GEO)
        for step in raw.ionic_steps:
            assert len(step["forces"]) == 7   # 7 atoms

    @needs_v4c3
    def test_v4c3_final_forces_small(self):
        # Converged geo-opt → forces should be small
        forces = vasprunrs.Vasprun(V4C3_GEO).ionic_steps[-1]["forces"]
        max_force = max(abs(f) for atom in forces for f in atom)
        assert max_force < 0.1   # eV/Å


class TestIncar:
    """INCAR dict has correct types and values."""

    def test_si_rpa_nbands(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        assert "NBANDS" in raw.incar

    def test_mos2_types(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        incar = raw.incar
        # LSORBIT should parse as bool
        if "LSORBIT" in incar:
            assert isinstance(incar["LSORBIT"], bool)

    @needs_pmg
    def test_si_rpa_parity(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        # Check a handful of scalar keys match
        for key in ("NBANDS", "ENCUT", "ISPIN"):
            if key in p.incar and key in v.incar:
                assert str(v.incar[key]) == str(p.incar[key]), f"INCAR[{key}] mismatch"


class TestKpoints:
    """Kpoint list, weights, and scheme are correct."""

    def test_si_rpa_weights_sum(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        assert abs(sum(raw.kpoints["weights"]) - 1.0) < 1e-6

    def test_mos2_weights_sum(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        assert abs(sum(raw.kpoints["weights"]) - 1.0) < 1e-6

    def test_si_rpa_kpointlist_shape(self):
        raw = vasprunrs.Vasprun(SI_RPA)
        kl = raw.kpoints["kpointlist"]
        assert kl.shape == (8, 3)

    @needs_pmg
    def test_si_rpa_kpoints_parity(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert np.allclose(v.actual_kpoints, p.actual_kpoints)
        assert np.allclose(v.actual_kpoints_weights, p.actual_kpoints_weights)


class TestInitialStructure:
    """initial_structure is parsed correctly."""

    def test_si_rpa_initial_equals_final(self):
        # Single-point — initial and final structures should be identical
        v = vrs(SI_RPA)
        assert len(v.initial_structure) == len(v.final_structure)
        assert abs(v.initial_structure.lattice.a - v.final_structure.lattice.a) < TOL

    @needs_v4c3
    def test_v4c3_initial_differs_from_final(self):
        # After relaxation the geometry changes
        v = vrs(V4C3_GEO)
        init_pos = v.initial_structure.frac_coords
        final_pos = v.final_structure.frac_coords
        assert not np.allclose(init_pos, final_pos, atol=1e-6)

    @needs_pmg
    def test_si_rpa_parity(self):
        v, p = vrs(SI_RPA), pmg(SI_RPA)
        assert abs(v.initial_structure.lattice.a - p.initial_structure.lattice.a) < TOL


class TestAtomTypes:
    """atom_types (mass, valence, pseudopotential) are exposed by the raw binding."""

    def test_mos2_atom_types(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        types = raw.atom_types
        syms = [t["element"] for t in types]
        assert "Mo" in syms and "S" in syms

    def test_mos2_mass(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        mo = next(t for t in raw.atom_types if t["element"] == "Mo")
        assert abs(mo["mass"] - 95.94) < 0.1

    def test_mos2_valence(self):
        raw = vasprunrs.Vasprun(MOS2_SOC)
        mo = next(t for t in raw.atom_types if t["element"] == "Mo")
        assert mo["valence"] > 0


# ===========================================================================
# projected_eigenvalues
# ===========================================================================
class TestProjectedEigenvalues:
    """projected_eigenvalues wiring: shape, values, and opt-in flag."""

    def test_none_when_not_requested(self):
        v = Vasprunrs(SI_RPA, parse_potcar_file=False, parse_projected_eigen=False)
        assert v.projected_eigenvalues is None

    @needs_mxene_h
    def test_shape_soc(self):
        v = Vasprunrs(MXENE_H_SOC, parse_potcar_file=False, parse_projected_eigen=True)
        pe = v.projected_eigenvalues
        assert pe is not None
        from pymatgen.electronic_structure.core import Spin
        # (nkpts=324, nbands=64, nions=10, norbitals=9)
        assert pe[Spin.up].shape == (324, 64, 10, 9)

    @needs_mxene_h
    def test_non_negative(self):
        v = Vasprunrs(MXENE_H_SOC, parse_potcar_file=False, parse_projected_eigen=True)
        from pymatgen.electronic_structure.core import Spin
        assert (v.projected_eigenvalues[Spin.up] >= 0).all()

    @needs_mxene_h
    @needs_pmg
    def test_parity_soc(self):
        v = Vasprunrs(MXENE_H_SOC, parse_potcar_file=False, parse_projected_eigen=True)
        p = pmg(MXENE_H_SOC)
        p._parse_projected_eigen = True
        from pymatgen.io.vasp import Vasprun as PV
        p2 = PV(MXENE_H_SOC, parse_potcar_file=False, parse_projected_eigen=True)
        from pymatgen.electronic_structure.core import Spin
        vd = v.projected_eigenvalues[Spin.up]
        pd = p2.projected_eigenvalues[Spin.up]
        assert np.abs(vd - pd).max() == 0.0

    @needs_v4c3
    @needs_pmg
    def test_parity_collinear(self):
        v = Vasprunrs(V4C3_GEO, parse_potcar_file=False, parse_projected_eigen=True)
        from pymatgen.io.vasp import Vasprun as PV
        p = PV(V4C3_GEO, parse_potcar_file=False, parse_projected_eigen=True)
        from pymatgen.electronic_structure.core import Spin
        vd = v.projected_eigenvalues[Spin.up]
        pd = p.projected_eigenvalues[Spin.up]
        assert np.abs(vd - pd).max() == 0.0
