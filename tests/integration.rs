use vasprunrs::{parse_file, ParseOptions};

// ── test fixtures ──────────────────────────────────────────────────────────
// Test-suite XMLs (committed to repo)
const SI_RPA:    &str = "tests/si_rpa.xml";
const N_GW:      &str = "tests/n_gw_large.xml";
const MOS2_SOC:  &str = "tests/mos2_soc.xml";

// Research XMLs — large real-world calculations (skipped if absent)
const COAGBI2TE3_SOC: &str = "/home/wladerer/research/interfaces/dosses/COAgBi2Te3/soc/vasprun.xml";
const AGBI2TE3_SOC:   &str = "/home/wladerer/research/interfaces/dosses/AgBi2Te3/soc/vasprun.xml";
const AUBI2TE3_BAND:  &str = "/home/wladerer/research/interfaces/dosses/AuBi2Te3/band/vasprun.xml";
const PT_SOC:         &str = "/home/wladerer/research/interfaces/dosses/slabs/Pt/bare/soc/vasprun.xml";

// ── helpers ────────────────────────────────────────────────────────────────

fn parse(path: &str) -> vasprunrs::Vasprun {
    parse_file(path, ParseOptions::default())
        .unwrap_or_else(|e| panic!("Failed to parse {path}: {e}"))
}

fn skip_if_absent(path: &str) -> bool {
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return true;
    }
    false
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

// ── Research files (skipped if not present) ────────────────────────────────

#[test]
fn research_coagbi2te3_soc() {
    if skip_if_absent(COAGBI2TE3_SOC) { return; }
    let v = parse(COAGBI2TE3_SOC);
    assert_structure_valid(&v.final_structure, "COAgBi2Te3");
    assert!(!v.atominfo.atoms.is_empty());
    assert_eigenvalues_valid(&v);
    eprintln!("COAgBi2Te3 SOC: {} atoms, {} kpts, {} ionic steps",
        v.atominfo.atoms.len(), v.kpoints.nkpts(), v.ionic_steps.len());
}

#[test]
fn research_agbi2te3_soc() {
    if skip_if_absent(AGBI2TE3_SOC) { return; }
    let v = parse(AGBI2TE3_SOC);
    assert_structure_valid(&v.final_structure, "AgBi2Te3");
    assert_eigenvalues_valid(&v);
    if let Some(dos) = &v.dos {
        assert!(!dos.total.energies.is_empty(), "DOS energies should be present");
    }
}

#[test]
fn research_aubi2te3_band() {
    if skip_if_absent(AUBI2TE3_BAND) { return; }
    let v = parse(AUBI2TE3_BAND);
    assert_structure_valid(&v.final_structure, "AuBi2Te3 band");
    assert_eigenvalues_valid(&v);
    // Band structure calculations typically use line-mode kpoints
    eprintln!("AuBi2Te3 band: {} kpts, labels={:?}",
        v.kpoints.nkpts(), v.kpoints.labels);
}

#[test]
fn research_pt_soc() {
    if skip_if_absent(PT_SOC) { return; }
    let v = parse(PT_SOC);
    assert_structure_valid(&v.final_structure, "Pt slab SOC");
    assert_eigenvalues_valid(&v);
    eprintln!("Pt SOC: {} atoms, {} kpts",
        v.atominfo.atoms.len(), v.kpoints.nkpts());
}
