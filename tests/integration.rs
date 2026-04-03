use vasprunrs::{parse_bytes, parse_file, ParseOptions};

// ── test fixtures (all gzip-compressed) ───────────────────────────────────
const SI_RPA:    &str = "tests/si_rpa.xml.gz";
const N_GW:      &str = "tests/n_gw_large.xml.gz";
const MOS2_SOC:  &str = "tests/mos2_soc.xml.gz";
const FE_FM:    &str = "tests/fe_fm.xml.gz";    // BCC Fe FM,  1 atom, NSW=0, LORBIT=11
const FE_AFM:   &str = "tests/fe_afm.xml.gz";   // BCC Fe AFM, 2 atoms, NSW=0, LORBIT=11
const FE_RELAX: &str = "tests/fe_relax.xml.gz"; // BCC Fe FM,  1 atom, NSW=5 (3 ionic steps)

// ── helpers ────────────────────────────────────────────────────────────────

fn parse(path: &str) -> vasprunrs::Vasprun {
    parse_file(path, ParseOptions::default())
        .unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"))
}

fn parse_gz(path: &str) -> vasprunrs::Vasprun { parse(path) }

fn parse_gz_opts(path: &str, opts: ParseOptions) -> vasprunrs::Vasprun {
    parse_file(path, opts)
        .unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"))
}

fn assert_structure_valid(s: &vasprunrs::Structure, label: &str) {
    assert!(!s.species.is_empty(), "{label}: species empty");
    assert_eq!(s.positions.len(), s.species.len(), "{label}: position/species count mismatch");
    assert_ne!(s.volume, 0.0, "{label}: volume is zero");
    // lattice vectors should be non-degenerate
    let det = {
        let a = s.lattice;
        a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
       -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
       +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
    };
    assert!(det.abs() > 1e-6, "{label}: degenerate lattice (det={det})");
}

fn assert_eigenvalues_valid(v: &vasprunrs::Vasprun) {
    if let Some(ref eig) = v.eigenvalues {
        assert!(eig.nspins >= 1 && eig.nspins <= 2, "nspins out of range");
        assert!(eig.nkpts > 0, "nkpts = 0");
        assert!(eig.nbands > 0, "nbands = 0");
        assert_eq!(eig.data.shape(), &[eig.nspins, eig.nkpts, eig.nbands, 2]);
        assert_eq!(eig.nkpts, v.kpoints.nkpts(), "eigenvalue kpt count ≠ kpoint list");
        // energies should be finite (NaN from overflow is allowed, but Inf is not)
        let data = &eig.data;
        for e in data.iter() {
            assert!(!e.is_infinite(), "infinite eigenvalue");
        }
    }
}

// ── Si RPA (503 KB, single-point, dielectric, DOS) ─────────────────────────

#[test]
fn si_rpa_basic() {
    let v = parse(SI_RPA);
    assert_eq!(v.atominfo.atoms, vec!["Si", "Si"]);
    assert_eq!(v.kpoints.nkpts(), 8);
    assert_structure_valid(&v.initial_structure, "initial");
    assert_structure_valid(&v.final_structure,   "final");
    assert_eq!(v.ionic_steps.len(), 1);
    assert!((v.ionic_steps[0].energy.e_fr_energy - (-11.690_705)).abs() < 1e-4);
}

#[test]
fn si_rpa_eigenvalues() {
    let v = parse(SI_RPA);
    assert_eigenvalues_valid(&v);
    let eig = v.eigenvalues.unwrap();
    assert_eq!((eig.nspins, eig.nkpts, eig.nbands), (1, 8, 48));
}

#[test]
fn si_rpa_dos() {
    let v = parse(SI_RPA);
    let dos = v.dos.expect("Si RPA should have DOS");
    assert_eq!(dos.total.energies.len(), 301);
    assert_eq!(dos.total.densities.shape(), &[1, 301]);
    assert!((dos.efermi - 6.326_521).abs() < 1e-4);
}

#[test]
fn si_rpa_dielectric() {
    let v = parse(SI_RPA);
    let diel = v.dielectric.expect("Si RPA should have dielectric");
    assert_eq!(diel.energies.len(), 1000);
    assert_eq!(diel.real.shape(), &[1000, 6]);
    assert_eq!(diel.imag.shape(), &[1000, 6]);
}

#[test]
fn si_rpa_incar() {
    use vasprunrs::IncarValue;
    let v = parse(SI_RPA);
    assert!(v.incar.contains_key("NBANDS"));
    assert!(matches!(v.incar.get("NBANDS"), Some(IncarValue::Int(_))));
}

// ── N GW (20 MB, GW post-processing, no ionic structure blocks) ─────────────

#[test]
fn n_gw_parses_without_ionic_steps() {
    let v = parse(N_GW);
    // GW calc blocks have no <structure> — ionic_steps should be empty, not an error
    assert_eq!(v.ionic_steps.len(), 0, "GW calc should yield 0 ionic steps");
    assert_structure_valid(&v.initial_structure, "initial");
    assert_structure_valid(&v.final_structure,   "final");
    assert_eq!(v.atominfo.atoms, vec!["N"]);
    assert_eigenvalues_valid(&v);
}

#[test]
fn n_gw_spin_polarized() {
    let v = parse(N_GW);
    let eig = v.eigenvalues.expect("GW should have eigenvalues");
    assert_eq!(eig.nspins, 2, "N GW is spin-polarized");
}

// ── MoS2 SOC (8.6 MB, float overflow `**` values) ──────────────────────────

#[test]
fn mos2_soc_overflow_values() {
    let v = parse(MOS2_SOC);
    assert_structure_valid(&v.final_structure, "MoS2 final");
    assert_eq!(v.atominfo.atoms, vec!["Mo", "S", "S"]);
    assert_eigenvalues_valid(&v);
    // The overflow NaN values should be in the data, not cause a crash
    if let Some(ref eig) = v.eigenvalues {
        let has_nan = eig.data.iter().any(|v| v.is_nan());
        // just a note — NaNs are acceptable, panics are not
        let _ = has_nan;
    }
}

#[test]
fn mos2_soc_kpoints() {
    let v = parse(MOS2_SOC);
    assert_eq!(v.kpoints.nkpts(), 29);
    // weights should sum to ~1.0
    let wsum: f64 = v.kpoints.weights.iter().sum();
    assert!((wsum - 1.0).abs() < 1e-6, "kpoint weights sum to {wsum}, expected 1.0");
}

// ── Magnetization ──────────────────────────────────────────────────────────

/// Minimal vasprun.xml snippet with a collinear magnetization block (ISPIN=2).
/// Each <v> in the magnetization varray has one float (total moment per atom).
const COLLINEAR_MAG_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<modeling>
 <generator>
  <i name="program" type="string">vasp</i>
  <i name="version" type="string">6.4.2</i>
  <i name="subversion" type="string"> (build)</i>
  <i name="platform" type="string">linux</i>
  <i name="date" type="string">2024 01 01</i>
  <i name="time" type="string">12:00:00</i>
 </generator>
 <incar><i type="int" name="ISPIN">2</i></incar>
 <atominfo>
  <atoms>2</atoms>
  <types>1</types>
  <array name="atoms">
   <dimension dim="1">ion</dimension>
   <field type="string">element</field>
   <field type="int">atomtype</field>
   <set>
    <rc><c>Fe</c><c>1</c></rc>
    <rc><c>Fe</c><c>1</c></rc>
   </set>
  </array>
  <array name="atomtypes">
   <dimension dim="1">type</dimension>
   <field type="int">atomspertype</field>
   <field type="string">element</field>
   <field>mass</field>
   <field>valence</field>
   <field type="string">pseudopotential</field>
   <set>
    <rc><c>2</c><c>Fe</c><c>55.845</c><c>8</c><c>PAW_PBE Fe</c></rc>
   </set>
  </array>
 </atominfo>
 <kpoints>
  <varray name="kpointlist"><v>0.0 0.0 0.0</v></varray>
  <varray name="weights"><v>1.0</v></varray>
 </kpoints>
 <structure name="initialpos">
  <crystal>
   <varray name="basis">
    <v>1.43 1.43 1.43</v>
    <v>-1.43 1.43 1.43</v>
    <v>-1.43 -1.43 1.43</v>
   </varray>
   <i name="volume">23.5</i>
   <varray name="rec_basis">
    <v>0.35 -0.35 0.0</v>
    <v>0.35 0.35 0.0</v>
    <v>0.0 0.0 0.35</v>
   </varray>
  </crystal>
  <varray name="positions">
   <v>0.0 0.0 0.0</v>
   <v>0.5 0.5 0.5</v>
  </varray>
 </structure>
 <calculation>
  <structure>
   <crystal>
    <varray name="basis">
     <v>1.43 1.43 1.43</v>
     <v>-1.43 1.43 1.43</v>
     <v>-1.43 -1.43 1.43</v>
    </varray>
    <i name="volume">23.5</i>
    <varray name="rec_basis">
     <v>0.35 -0.35 0.0</v>
     <v>0.35 0.35 0.0</v>
     <v>0.0 0.0 0.35</v>
    </varray>
   </crystal>
   <varray name="positions">
    <v>0.0 0.0 0.0</v>
    <v>0.5 0.5 0.5</v>
   </varray>
  </structure>
  <varray name="forces">
   <v>0.0 0.0 0.0</v>
   <v>0.0 0.0 0.0</v>
  </varray>
  <varray name="stress">
   <v>1.0 0.0 0.0</v>
   <v>0.0 1.0 0.0</v>
   <v>0.0 0.0 1.0</v>
  </varray>
  <varray name="magnetization">
   <v>2.1</v>
   <v>-2.1</v>
  </varray>
  <energy>
   <i name="e_fr_energy">-16.2</i>
   <i name="e_wo_entrp">-16.1</i>
   <i name="e_0_energy">-16.15</i>
  </energy>
 </calculation>
 <structure name="finalpos">
  <crystal>
   <varray name="basis">
    <v>1.43 1.43 1.43</v>
    <v>-1.43 1.43 1.43</v>
    <v>-1.43 -1.43 1.43</v>
   </varray>
   <i name="volume">23.5</i>
   <varray name="rec_basis">
    <v>0.35 -0.35 0.0</v>
    <v>0.35 0.35 0.0</v>
    <v>0.0 0.0 0.35</v>
   </varray>
  </crystal>
  <varray name="positions">
   <v>0.0 0.0 0.0</v>
   <v>0.5 0.5 0.5</v>
  </varray>
 </structure>
</modeling>"#;

#[test]
fn magnetization_absent_for_non_magnetic() {
    // Si RPA is non-magnetic — magnetization should be None on every ionic step
    let v = parse(SI_RPA);
    for step in &v.ionic_steps {
        assert!(
            step.magnetization.is_none(),
            "non-magnetic calculation should have no magnetization",
        );
    }
}

#[test]
fn magnetization_collinear_parsed() {
    let v = parse_bytes(COLLINEAR_MAG_XML.as_bytes(), ParseOptions::default()).unwrap();
    assert_eq!(v.ionic_steps.len(), 1);
    let mag = v.ionic_steps[0].magnetization.as_ref()
        .expect("magnetization should be present for ISPIN=2 calc");
    assert_eq!(mag.len(), 2, "one entry per atom");
    // collinear: each entry is a single float
    assert_eq!(mag[0].len(), 1);
    assert_eq!(mag[1].len(), 1);
    assert!((mag[0][0] - 2.1).abs() < 1e-9,  "Fe[0] moment ≈ 2.1 μB");
    assert!((mag[1][0] - (-2.1)).abs() < 1e-9, "Fe[1] moment ≈ -2.1 μB");
}


// ── Fe fixtures (VASP 6.4.1, gzip-compressed) ─────────────────────────────
// Reference values cross-checked against pymatgen.io.vasp.Vasprun.

#[test]
fn fe_fm_basic() {
    let v = parse_gz(FE_FM);
    assert_eq!(v.atominfo.atoms, vec!["Fe"]);
    assert_eq!(v.kpoints.nkpts(), 29);
    assert_eq!(v.ionic_steps.len(), 1);
    assert!((v.ionic_steps[0].energy.e_fr_energy - (-8.227_466)).abs() < 1e-4);
    assert_structure_valid(&v.final_structure, "Fe FM final");
}

#[test]
fn fe_fm_spin_polarized_eigenvalues() {
    let v = parse_gz(FE_FM);
    let eig = v.eigenvalues.as_ref().expect("Fe FM should have eigenvalues");
    // ISPIN=2 → 2 spin channels; 29 irreducible kpoints; 9 bands
    assert_eq!((eig.nspins, eig.nkpts, eig.nbands), (2, 29, 9));
    assert_eigenvalues_valid(&v);
}

#[test]
fn fe_fm_efermi() {
    let v = parse_gz(FE_FM);
    let efermi = v.efermi.expect("Fe FM should have efermi");
    assert!((efermi - 5.857_667).abs() < 1e-4);
}

#[test]
fn fe_fm_magnetization_absent_vasp6() {
    // VASP 6.x does not write <varray name="magnetization"> in the
    // calculation block.  The field should be None for all ionic steps.
    let v = parse_gz(FE_FM);
    for step in &v.ionic_steps {
        assert!(step.magnetization.is_none(),
            "VASP 6.x should not produce a magnetization varray");
    }
}

#[test]
fn fe_afm_two_atoms() {
    let v = parse_gz(FE_AFM);
    assert_eq!(v.atominfo.atoms, vec!["Fe", "Fe"]);
    assert_eq!(v.kpoints.nkpts(), 20);
    assert_eq!(v.ionic_steps.len(), 1);
    assert!((v.ionic_steps[0].energy.e_fr_energy - (-15.592_996)).abs() < 1e-4);
    let eig = v.eigenvalues.expect("Fe AFM should have eigenvalues");
    assert_eq!((eig.nspins, eig.nkpts, eig.nbands), (2, 20, 12));
}

#[test]
fn fe_relax_ionic_steps() {
    let v = parse_gz(FE_RELAX);
    assert_eq!(v.ionic_steps.len(), 3);
    // energies should decrease (or stay flat) over the relaxation
    let energies: Vec<f64> = v.ionic_steps.iter()
        .map(|s| s.energy.e_fr_energy)
        .collect();
    for w in energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-6,
            "energy should not increase during relaxation: {:.6} → {:.6}", w[0], w[1]);
    }
    assert!((v.ionic_steps.last().unwrap().energy.e_fr_energy - (-8.229_077)).abs() < 1e-4);
}
