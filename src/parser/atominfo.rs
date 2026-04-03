use std::collections::HashMap;
use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::{AtomInfo, AtomType};
use super::helpers::*;

/// Parse <atominfo> block. Called after Start("atominfo") event.
pub fn parse_atominfo(reader: &mut XmlReader) -> Result<AtomInfo> {
    let mut atoms: Option<Vec<String>> = None;
    let mut atom_types: Option<Vec<AtomType>> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"array" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        match name_attr.as_str() {
                            "atoms" => {
                                atoms = Some(parse_atoms_array(reader)?);
                            }
                            "atomtypes" => {
                                atom_types = Some(parse_atomtypes_array(reader)?);
                            }
                            _ => {
                                skip_element(reader, b"array")?;
                            }
                        }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"atominfo" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let atoms = atoms.ok_or_else(|| VasprunError::MissingElement("array[atoms]".into()))?;
    let atom_types =
        atom_types.ok_or_else(|| VasprunError::MissingElement("array[atomtypes]".into()))?;

    Ok(AtomInfo { atoms, atom_types })
}

/// Parse <array name="atoms"> block. Called after Start("array") event consumed.
/// Reads until </array>.
fn parse_atoms_array(reader: &mut XmlReader) -> Result<Vec<String>> {
    // We need to find the <set> inside this array.
    // Skip <dimension> and <field>, then parse <set>.
    let mut atoms = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"set" => {
                        atoms = parse_atoms_set(reader)?;
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"array" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    Ok(atoms)
}

/// Parse the <set> inside atoms array. Called after Start("set") event.
fn parse_atoms_set(reader: &mut XmlReader) -> Result<Vec<String>> {
    let mut atoms = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"rc" => {
                // Each <rc><c>Si</c><c>1</c></rc>
                let element = parse_rc_first_c(reader)?;
                atoms.push(element.trim().to_string());
            }
            Event::End(ref e) if e.name().as_ref() == b"set" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(atoms)
}

/// Parse <rc> block, returning text of first <c>. Reads until </rc>.
fn parse_rc_first_c(reader: &mut XmlReader) -> Result<String> {
    let mut first_c: Option<String> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"c" => {
                let text = read_text(reader)?;
                if first_c.is_none() {
                    first_c = Some(text);
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"rc" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(first_c.unwrap_or_default())
}

/// Parse <rc> block, returning all <c> text values. Reads until </rc>.
fn parse_rc_all_c(reader: &mut XmlReader) -> Result<Vec<String>> {
    let mut cols = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"c" => {
                let text = read_text(reader)?;
                cols.push(text);
            }
            Event::End(ref e) if e.name().as_ref() == b"rc" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(cols)
}

/// Parse <array name="atomtypes"> block. Called after Start("array") event consumed.
fn parse_atomtypes_array(reader: &mut XmlReader) -> Result<Vec<AtomType>> {
    let mut fields = Vec::new();
    let mut types = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"field" => {
                        let text = read_text(reader)?;
                        fields.push(text);
                    }
                    b"set" => {
                        // Build field_index map
                        let field_index: HashMap<&str, usize> = fields
                            .iter()
                            .enumerate()
                            .map(|(i, f)| (f.as_str(), i))
                            .collect();
                        types = parse_atomtypes_set(reader, &fields, &field_index)?;
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"array" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    Ok(types)
}

fn parse_atomtypes_set(
    reader: &mut XmlReader,
    _fields: &[String],
    field_index: &HashMap<&str, usize>,
) -> Result<Vec<AtomType>> {
    let mut types = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"rc" => {
                let cols = parse_rc_all_c(reader)?;

                let get = |name: &str| -> &str {
                    field_index
                        .get(name)
                        .and_then(|&i| cols.get(i))
                        .map(|s| s.as_str())
                        .unwrap_or("")
                };

                let count = get("atomspertype")
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| VasprunError::ParseValue {
                        value: get("atomspertype").to_string(),
                        target: "usize".into(),
                        reason: "atomspertype is not a valid integer".into(),
                    })?;
                let mass = get("mass")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| VasprunError::ParseValue {
                        value: get("mass").to_string(),
                        target: "f64".into(),
                        reason: "mass is not a valid float".into(),
                    })?;
                let valence = get("valence")
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| VasprunError::ParseValue {
                        value: get("valence").to_string(),
                        target: "f64".into(),
                        reason: "valence is not a valid float".into(),
                    })?;

                types.push(AtomType {
                    element:         get("element").to_string(),
                    count,
                    mass,
                    valence,
                    pseudopotential: get("pseudopotential").to_string(),
                });
            }
            Event::End(ref e) if e.name().as_ref() == b"set" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(types)
}
