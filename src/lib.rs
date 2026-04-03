pub mod error;
pub mod types;
pub mod parser;
#[cfg(feature = "python")]
mod python;

pub use error::{Result, VasprunError};
pub use types::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python extension module. Registered by maturin when built with `--features python`.
#[cfg(feature = "python")]
#[pymodule]
fn vasprunrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}

use std::path::Path;

/// Options controlling what gets parsed.
///
/// Mirrors `pymatgen.io.vasp.outputs.Vasprun.__init__` keyword arguments so the
/// two APIs stay in sync.
#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// Parse eigenvalues (default: true). Disable to speed up structure/energy-only reads.
    pub parse_eigen: bool,
    /// Parse projected eigenvalues (LORBIT >= 10). Large; off by default.
    pub parse_projected: bool,
    /// Parse the DOS block (default: true). Disable for non-DOS calculations.
    pub parse_dos: bool,
    /// Only keep every Nth ionic step (None = keep all).
    pub ionic_step_skip: Option<usize>,
    /// Skip the first N ionic steps before applying `ionic_step_skip`.
    pub ionic_step_offset: usize,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            parse_eigen: true,
            parse_projected: false,
            parse_dos: true,
            ionic_step_skip: None,
            ionic_step_offset: 0,
        }
    }
}

/// Parse a `vasprun.xml` (or `vasprun.xml.gz`) file from disk.
///
/// Files whose name ends with `.gz` are decompressed transparently.
pub fn parse_file(path: impl AsRef<Path>, opts: ParseOptions) -> Result<Vasprun> {
    let path = path.as_ref();
    let raw = if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        use flate2::read::GzDecoder;
        use std::io::Read;
        let file = std::fs::File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)
            .map_err(|e| VasprunError::Io(e))?;
        buf
    } else {
        std::fs::read(path)?
    };
    parse_bytes(&raw, opts)
}

/// Parse a `vasprun.xml` from an in-memory byte slice.
pub fn parse_bytes(raw: &[u8], opts: ParseOptions) -> Result<Vasprun> {
    let xml = transcode_to_utf8(raw)?;
    let mut parser = parser::VasprunParser::new(xml.as_bytes());
    parser::parse_document(&mut parser, &opts)
}

/// Transcode ISO-8859-1 (or plain UTF-8) XML bytes to a UTF-8 String.
///
/// quick-xml requires UTF-8. vasprun.xml declares ISO-8859-1 but in practice
/// the content is ASCII-compatible; encoding_rs handles both transparently.
fn transcode_to_utf8(raw: &[u8]) -> Result<String> {
    // Fast path: already valid UTF-8 → avoid a copy.
    if let Ok(s) = std::str::from_utf8(raw) {
        return Ok(rewrite_xml_declaration(s));
    }

    // Slow path: decode as ISO-8859-1.
    let (cow, _, had_errors) = encoding_rs::WINDOWS_1252.decode(raw);
    if had_errors {
        return Err(VasprunError::Encoding(
            "Failed to decode vasprun.xml as ISO-8859-1".into(),
        ));
    }
    Ok(rewrite_xml_declaration(&cow))
}

/// Replace `encoding="ISO-8859-1"` with `encoding="UTF-8"` so quick-xml accepts the document.
fn rewrite_xml_declaration(s: &str) -> String {
    if let Some(pos) = s.find("?>") {
        let (decl, rest) = s.split_at(pos + 2);
        let fixed = decl.replace("ISO-8859-1", "UTF-8")
                        .replace("iso-8859-1", "UTF-8");
        return format!("{fixed}{rest}");
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<modeling>
 <generator>
  <i name="program" type="string">vasp</i>
  <i name="version" type="string">6.4.1</i>
  <i name="subversion" type="string"> (build)</i>
  <i name="platform" type="string">linux</i>
  <i name="date" type="string">2024 01 01</i>
  <i name="time" type="string">12:00:00</i>
 </generator>
 <incar>
  <i type="int"     name="NBANDS">8</i>
  <i type="logical" name="LREAL">F</i>
  <i                name="ENCUT">400.0</i>
 </incar>
 <atominfo>
  <atoms>2</atoms>
  <types>1</types>
  <array name="atoms">
   <dimension dim="1">ion</dimension>
   <field type="string">element</field>
   <field type="int">atomtype</field>
   <set>
    <rc><c>Si</c><c>1</c></rc>
    <rc><c>Si</c><c>1</c></rc>
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
    <rc><c>2</c><c>Si</c><c>28.085</c><c>4</c><c>PAW_PBE Si</c></rc>
   </set>
  </array>
 </atominfo>
 <kpoints>
  <generation param="Gamma">
   <v type="int" name="divisions">4 4 4</v>
   <v name="usershift">0.0 0.0 0.0</v>
  </generation>
  <varray name="kpointlist">
   <v>0.0 0.0 0.0</v>
   <v>0.25 0.0 0.0</v>
  </varray>
  <varray name="weights">
   <v>0.5</v>
   <v>0.5</v>
  </varray>
 </kpoints>
 <structure name="initialpos">
  <crystal>
   <varray name="basis">
    <v>2.71 2.71 0.0</v>
    <v>0.0  2.71 2.71</v>
    <v>2.71 0.0  2.71</v>
   </varray>
   <i name="volume">39.9</i>
   <varray name="rec_basis">
    <v>0.184 0.184 -0.184</v>
    <v>-0.184 0.184 0.184</v>
    <v>0.184 -0.184 0.184</v>
   </varray>
  </crystal>
  <varray name="positions">
   <v>0.0 0.0 0.0</v>
   <v>0.25 0.25 0.25</v>
  </varray>
 </structure>
 <calculation>
  <scstep>
   <energy>
    <i name="e_fr_energy">-10.5</i>
    <i name="e_wo_entrp">-10.4</i>
    <i name="e_0_energy">-10.45</i>
   </energy>
  </scstep>
  <structure>
   <crystal>
    <varray name="basis">
     <v>2.71 2.71 0.0</v>
     <v>0.0  2.71 2.71</v>
     <v>2.71 0.0  2.71</v>
    </varray>
    <i name="volume">39.9</i>
    <varray name="rec_basis">
     <v>0.184 0.184 -0.184</v>
     <v>-0.184 0.184 0.184</v>
     <v>0.184 -0.184 0.184</v>
    </varray>
   </crystal>
   <varray name="positions">
    <v>0.0 0.0 0.0</v>
    <v>0.25 0.25 0.25</v>
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
  <energy>
   <i name="e_fr_energy">-10.5</i>
   <i name="e_wo_entrp">-10.4</i>
   <i name="e_0_energy">-10.45</i>
  </energy>
 </calculation>
 <structure name="finalpos">
  <crystal>
   <varray name="basis">
    <v>2.71 2.71 0.0</v>
    <v>0.0  2.71 2.71</v>
    <v>2.71 0.0  2.71</v>
   </varray>
   <i name="volume">39.9</i>
   <varray name="rec_basis">
    <v>0.184 0.184 -0.184</v>
    <v>-0.184 0.184 0.184</v>
    <v>0.184 -0.184 0.184</v>
   </varray>
  </crystal>
  <varray name="positions">
   <v>0.0 0.0 0.0</v>
   <v>0.25 0.25 0.25</v>
  </varray>
 </structure>
</modeling>"#;

    #[test]
    fn test_minimal_parse() {
        let v = parse_bytes(SAMPLE_XML.as_bytes(), ParseOptions::default()).unwrap();
        assert_eq!(v.generator.program, "vasp");
        assert_eq!(v.atominfo.atoms, vec!["Si", "Si"]);
        assert_eq!(v.kpoints.nkpts(), 2);
        assert_eq!(v.initial_structure.natoms(), 2);
        assert_eq!(v.ionic_steps.len(), 1);
        assert!((v.ionic_steps[0].energy.e_fr_energy - (-10.5)).abs() < 1e-9);
    }

    #[test]
    fn test_incar_types() {
        let v = parse_bytes(SAMPLE_XML.as_bytes(), ParseOptions::default()).unwrap();
        assert!(matches!(v.incar["NBANDS"], IncarValue::Int(8)));
        assert!(matches!(v.incar["LREAL"], IncarValue::Bool(false)));
        assert!(matches!(v.incar["ENCUT"], IncarValue::Float(_)));
    }
}
