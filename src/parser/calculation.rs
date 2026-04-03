use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::{IonicStep, ScfEnergy, Eigenvalues, Projected, Dos, Dielectric};
use crate::ParseOptions;
use super::helpers::*;
use super::kpoints::{parse_varray_v3, parse_varray_floats};
use super::structure::parse_structure;

/// Result of parsing a single <calculation> block.
pub struct CalcResult {
    pub ionic_step: Option<IonicStep>,
    pub eigenvalues: Option<Eigenvalues>,
    pub projected: Option<Projected>,
    pub dos: Option<Dos>,
    pub dielectric: Option<Dielectric>,
}

/// Parse a single <calculation> block. Called after Start("calculation") event.
/// Reads until </calculation>.
pub fn parse_calculation(
    reader: &mut XmlReader,
    atoms: &[String],
    opts: &ParseOptions,
) -> Result<CalcResult> {
    let mut scf_steps: Vec<ScfEnergy> = Vec::new();
    let mut structure: Option<crate::types::Structure> = None;
    let mut forces: Vec<[f64; 3]> = Vec::new();
    let mut stress_rows: Vec<[f64; 3]> = Vec::new();
    let mut magnetization: Option<Vec<Vec<f64>>> = None;
    let mut energy = ScfEnergy::default();
    let mut eigenvalues: Option<Eigenvalues> = None;
    let mut projected: Option<Projected> = None;
    let mut dos: Option<Dos> = None;
    let mut dielectric: Option<Dielectric> = None;
    let mut has_structure = false;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"scstep" => {
                        let scf = parse_scstep(reader)?;
                        scf_steps.push(scf);
                    }
                    b"structure" => {
                        let s = parse_structure(reader, atoms)?;
                        has_structure = true;
                        structure = Some(s);
                    }
                    b"varray" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        match name_attr.as_str() {
                            "forces" => {
                                forces = parse_varray_v3(reader)?;
                            }
                            "stress" => {
                                stress_rows = parse_varray_v3(reader)?;
                            }
                            "magnetization" => {
                                magnetization = Some(parse_varray_floats(reader)?);
                            }
                            _ => {
                                skip_element(reader, b"varray")?;
                            }
                        }
                    }
                    b"energy" => {
                        energy = parse_energy_fields(reader)?;
                    }
                    b"eigenvalues" => {
                        if opts.parse_eigen {
                            eigenvalues = Some(super::eigenvalues::parse_eigenvalues(reader)?);
                        } else {
                            reader.fast_skip(b"eigenvalues")?;
                        }
                    }
                    b"projected" => {
                        if opts.parse_projected {
                            projected = Some(super::eigenvalues::parse_projected(reader)?);
                        } else {
                            reader.fast_skip(b"projected")?;
                        }
                    }
                    b"dos" => {
                        if opts.parse_dos {
                            dos = Some(super::dos::parse_dos(reader)?);
                        } else {
                            reader.fast_skip(b"dos")?;
                        }
                    }
                    b"dielectricfunction" => {
                        // Take the first block only (density-density); VASP RPA writes a
                        // second current-current block that pymatgen also ignores.
                        if dielectric.is_none() {
                            dielectric = Some(super::dielectric::parse_dielectric(reader)?);
                        } else {
                            reader.fast_skip(b"dielectricfunction")?;
                        }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"calculation" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let ionic_step = if has_structure {
        let structure = structure.ok_or_else(|| {
            VasprunError::MissingElement("structure inside calculation".into())
        })?;

        let stress = if stress_rows.len() >= 3 {
            [stress_rows[0], stress_rows[1], stress_rows[2]]
        } else {
            [[0.0; 3]; 3]
        };

        Some(IonicStep {
            structure,
            forces,
            stress,
            energy,
            scf_steps,
            magnetization,
        })
    } else {
        None
    };

    Ok(CalcResult {
        ionic_step,
        eigenvalues,
        projected,
        dos,
        dielectric,
    })
}

/// Parse <energy> block. Called after Start("energy") event. Reads until </energy>.
pub fn parse_energy_fields(reader: &mut XmlReader) -> Result<ScfEnergy> {
    let mut e = ScfEnergy::default();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref ev) if ev.name().as_ref() == b"i" => {
                let name_attr = attr_str(ev, b"name").unwrap_or_default();
                let text = read_text(reader)?;
                let val: f64 = text.parse().unwrap_or(0.0);
                match name_attr.as_str() {
                    "e_fr_energy" => e.e_fr_energy = val,
                    "e_wo_entrp"  => e.e_wo_entrp  = val,
                    "e_0_energy"  => e.e_0_energy  = val,
                    _ => {}
                }
            }
            Event::End(ref ev) if ev.name().as_ref() == b"energy" => break,
            Event::Eof => break,
            Event::Start(ref ev) => {
                let tag = ev.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(e)
}

/// Parse <scstep> block. Called after Start("scstep") event. Reads until </scstep>.
fn parse_scstep(reader: &mut XmlReader) -> Result<ScfEnergy> {
    let mut scf = ScfEnergy::default();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"energy" => {
                scf = parse_energy_fields(reader)?;
            }
            Event::End(ref e) if e.name().as_ref() == b"scstep" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(scf)
}
