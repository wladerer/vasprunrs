mod helpers;
pub mod incar;
pub mod atominfo;
pub mod kpoints;
pub mod structure;
pub mod calculation;
pub mod eigenvalues;
pub mod dos;
pub mod dielectric;

pub use helpers::*;

use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::*;
use crate::ParseOptions;

/// Parse a full vasprun.xml document into a [`Vasprun`].
pub fn parse_document(reader: &mut helpers::XmlReader, opts: &ParseOptions) -> Result<Vasprun> {
    // Scan for <modeling>
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"modeling" => break,
            Event::Eof => return Err(VasprunError::MissingElement("modeling".into())),
            _ => {}
        }
    }

    // State to accumulate
    let mut generator: Option<Generator> = None;
    let mut incar: Option<indexmap::IndexMap<String, IncarValue>> = None;
    let mut atominfo_opt: Option<AtomInfo> = None;
    let mut kpoints_opt: Option<Kpoints> = None;
    let mut initial_structure: Option<Structure> = None;
    let mut final_structure: Option<Structure> = None;
    let mut all_ionic_steps: Vec<IonicStep> = Vec::new();
    let mut eig_opt: Option<Eigenvalues> = None;
    let mut proj_opt: Option<Projected> = None;
    let mut dos_opt: Option<Dos> = None;
    let mut diel_opt: Option<Dielectric> = None;

    // Parse children of <modeling>
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"generator" => {
                        generator = Some(incar::parse_generator(reader)?);
                    }
                    b"incar" => {
                        incar = Some(incar::parse_incar(reader)?);
                    }
                    b"atominfo" => {
                        atominfo_opt = Some(atominfo::parse_atominfo(reader)?);
                    }
                    b"kpoints" => {
                        kpoints_opt = Some(kpoints::parse_kpoints(reader)?);
                    }
                    b"structure" => {
                        // Determine if initialpos or finalpos by name attribute
                        let name_attr = attr_str(e, b"name");
                        let atoms = atominfo_opt
                            .as_ref()
                            .map(|a| a.atoms.as_slice())
                            .unwrap_or(&[]);
                        let s = structure::parse_structure(reader, atoms)?;
                        match name_attr.as_deref() {
                            Some("initialpos") | None => {
                                if initial_structure.is_none() {
                                    initial_structure = Some(s);
                                } else {
                                    final_structure = Some(s);
                                }
                            }
                            Some("finalpos") => {
                                final_structure = Some(s);
                            }
                            Some(_) => {
                                // Any other named structure — treat as initial if not set
                                if initial_structure.is_none() {
                                    initial_structure = Some(s);
                                } else {
                                    final_structure = Some(s);
                                }
                            }
                        }
                    }
                    b"calculation" => {
                        let atoms = atominfo_opt
                            .as_ref()
                            .map(|a| a.atoms.as_slice())
                            .unwrap_or(&[]);
                        let result = calculation::parse_calculation(
                            reader, atoms, opts,
                        )?;
                        if let Some(step) = result.ionic_step {
                            all_ionic_steps.push(step);
                        }
                        // Always keep the latest eigenvalues/dos/dielectric
                        if let Some(e) = result.eigenvalues { eig_opt = Some(e); }
                        if let Some(p) = result.projected { proj_opt = Some(p); }
                        if let Some(d) = result.dos { dos_opt = Some(d); }
                        if let Some(d) = result.dielectric { diel_opt = Some(d); }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"modeling" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let generator = generator.ok_or_else(|| VasprunError::MissingElement("generator".into()))?;
    let incar = incar.ok_or_else(|| VasprunError::MissingElement("incar".into()))?;
    let atominfo = atominfo_opt.ok_or_else(|| VasprunError::MissingElement("atominfo".into()))?;
    let kpoints = kpoints_opt.ok_or_else(|| VasprunError::MissingElement("kpoints".into()))?;

    let initial_structure = initial_structure
        .ok_or_else(|| VasprunError::MissingElement("structure[initialpos]".into()))?;

    // final_structure falls back to initial if not found
    let final_structure = final_structure.unwrap_or_else(|| initial_structure.clone());

    // Apply ionic_step_offset and ionic_step_skip
    let ionic_steps = {
        let offset = opts.ionic_step_offset;
        let after_offset = if offset < all_ionic_steps.len() {
            all_ionic_steps.drain(offset..).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        match opts.ionic_step_skip {
            None | Some(0) => after_offset,
            Some(skip) => after_offset.into_iter().step_by(skip).collect(),
        }
    };

    let eigenvalues = if opts.parse_eigen { eig_opt } else { None };
    let projected = if opts.parse_projected { proj_opt } else { None };
    let dos = if opts.parse_dos { dos_opt } else { None };
    let efermi = dos.as_ref().map(|d| d.efermi);
    let dielectric = diel_opt;

    Ok(Vasprun {
        generator,
        incar,
        atominfo,
        kpoints,
        initial_structure,
        final_structure,
        ionic_steps,
        eigenvalues,
        projected,
        dos,
        dielectric,
        efermi,
    })
}
