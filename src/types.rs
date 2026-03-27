use indexmap::IndexMap;
use ndarray::{Array2, Array4, Array5};

/// Top-level parsed representation of a vasprun.xml file.
#[derive(Debug)]
pub struct Vasprun {
    pub generator:         Generator,
    pub incar:             IndexMap<String, IncarValue>,
    pub atominfo:          AtomInfo,
    pub kpoints:           Kpoints,
    pub initial_structure: Structure,
    pub final_structure:   Structure,
    pub ionic_steps:       Vec<IonicStep>,
    pub eigenvalues:       Option<Eigenvalues>,
    pub projected:         Option<Projected>,
    pub dos:               Option<Dos>,
    pub dielectric:        Option<Dielectric>,
    pub efermi:            Option<f64>,
}

#[derive(Debug, Default)]
pub struct Generator {
    pub program:    String,
    pub version:    String,
    pub subversion: String,
    pub platform:   String,
    pub date:       String,
    pub time:       String,
}

/// Scalar or vector INCAR value.
#[derive(Debug, Clone)]
pub enum IncarValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Vec(Vec<f64>),
}

impl std::fmt::Display for IncarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v)   => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(v)  => write!(f, "{v}"),
            Self::Str(v)   => write!(f, "{v}"),
            Self::Vec(v)   => write!(f, "{v:?}"),
        }
    }
}

/// Atomic structure: lattice + per-atom species and fractional coords.
#[derive(Debug, Clone)]
pub struct Structure {
    /// Row-major: basis[i] is the i-th lattice vector in Angstrom.
    pub lattice:   [[f64; 3]; 3],
    pub rec_basis: [[f64; 3]; 3],
    pub volume:    f64,
    /// Element symbol per atom (length = natoms).
    pub species:   Vec<String>,
    /// Fractional coordinates per atom.
    pub positions: Vec<[f64; 3]>,
}

impl Structure {
    pub fn natoms(&self) -> usize { self.positions.len() }
}

/// Per-species metadata from <atomtypes>.
#[derive(Debug, Clone)]
pub struct AtomType {
    pub element:         String,
    pub count:           usize,
    pub mass:            f64,
    pub valence:         f64,
    pub pseudopotential: String,
}

#[derive(Debug)]
pub struct AtomInfo {
    /// One entry per atom (element symbol).
    pub atoms:      Vec<String>,
    /// One entry per species.
    pub atom_types: Vec<AtomType>,
}

#[derive(Debug)]
pub struct KpointGeneration {
    pub scheme:     String,             // "Gamma", "Monkhorst-Pack", "Line-mode", etc.
    pub divisions:  Option<[i32; 3]>,
    pub usershift:  [f64; 3],
}

#[derive(Debug)]
pub struct Kpoints {
    pub generation: Option<KpointGeneration>,
    /// Fractional coordinates in reciprocal space.
    pub kpointlist: Array2<f64>,        // shape [nkpts, 3]
    pub weights:    Vec<f64>,
    /// High-symmetry point labels, if present (line-mode calculations).
    pub labels:     Vec<(usize, String)>, // (kpt_index, label)
}

impl Kpoints {
    pub fn nkpts(&self) -> usize { self.weights.len() }
}

/// Energy contributions for a single SCF step.
#[derive(Debug, Clone, Default)]
pub struct ScfEnergy {
    pub e_fr_energy: f64,
    pub e_wo_entrp:  f64,
    pub e_0_energy:  f64,
}

/// One ionic (geometry) step, containing all SCF steps and final structural data.
#[derive(Debug)]
pub struct IonicStep {
    pub structure:   Structure,
    pub forces:      Vec<[f64; 3]>,
    pub stress:      [[f64; 3]; 3],    // kBar
    pub energy:      ScfEnergy,        // final converged energy
    pub scf_steps:   Vec<ScfEnergy>,
}

/// Kohn-Sham eigenvalues.
/// `data` shape: [nspins, nkpts, nbands, 2]  — axis-3 is [energy, occupancy].
#[derive(Debug)]
pub struct Eigenvalues {
    pub nspins: usize,
    pub nkpts:  usize,
    pub nbands: usize,
    pub data:   Array4<f64>,
}

/// Projected eigenvalues (LORBIT ≥ 10).
/// `data` shape: [nspins, nkpts, nbands, nions, norbitals]
#[derive(Debug)]
pub struct Projected {
    pub nspins:    usize,
    pub nkpts:     usize,
    pub nbands:    usize,
    pub nions:     usize,
    pub norbitals: usize,
    pub data:      Array5<f64>,
    /// Orbital labels in order (e.g. ["s","py","pz","px","dxy",...])
    pub orbitals:  Vec<String>,
}

/// Total and partial (site + orbital) DOS.
#[derive(Debug)]
pub struct Dos {
    pub efermi:  f64,
    pub total:   DosData,
    pub partial: Option<PartialDos>,
}

/// DOS on a regular energy grid.
/// `densities` shape: [nspins, nedos]
#[derive(Debug)]
pub struct DosData {
    pub energies:   Vec<f64>,
    pub densities:  Array2<f64>,    // [nspins, nedos]
    pub integrated: Array2<f64>,    // [nspins, nedos]
}

/// Partial DOS projected onto ions and orbitals.
/// `data` shape: [nspins, nions, norbitals, nedos]
#[derive(Debug)]
pub struct PartialDos {
    pub data:     ndarray::Array4<f64>,
    pub orbitals: Vec<String>,
}

/// Dielectric function (real + imaginary parts).
/// Each component shape: [nfreq, 6]  — columns are xx yy zz xy yz zx.
#[derive(Debug)]
pub struct Dielectric {
    pub energies: Vec<f64>,
    pub real:     Array2<f64>,   // [nfreq, 6]
    pub imag:     Array2<f64>,   // [nfreq, 6]
}
