use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::Structure;
use super::helpers::*;
use super::kpoints::parse_varray_v3;

/// Parse <structure> block. Called after Start("structure") event.
/// Reads until </structure>.
pub fn parse_structure(reader: &mut XmlReader, atoms: &[String]) -> Result<Structure> {
    let mut lattice: Option<[[f64; 3]; 3]> = None;
    let mut rec_basis: Option<[[f64; 3]; 3]> = None;
    let mut volume = 0.0f64;
    let mut positions: Option<Vec<[f64; 3]>> = None;
    let mut selective: Option<Vec<[bool; 3]>> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"crystal" => {
                        let (lat, rb, vol) = parse_crystal(reader)?;
                        lattice = Some(lat);
                        rec_basis = Some(rb);
                        volume = vol;
                    }
                    b"varray" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        match name_attr.as_str() {
                            "positions" => {
                                positions = Some(parse_varray_v3(reader)?);
                            }
                            "selective" => {
                                selective = Some(parse_selective(reader)?);
                            }
                            _ => {
                                skip_element(reader, b"varray")?;
                            }
                        }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"structure" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let lattice = lattice.ok_or_else(|| VasprunError::MissingElement("crystal".into()))?;
    let rec_basis = rec_basis.ok_or_else(|| VasprunError::MissingElement("varray[rec_basis]".into()))?;
    let positions = positions.ok_or_else(|| VasprunError::MissingElement("varray[positions]".into()))?;

    if !atoms.is_empty() && positions.len() != atoms.len() {
        return Err(VasprunError::ShapeMismatch {
            expected: vec![atoms.len()],
            got: vec![positions.len()],
        });
    }

    let species = if atoms.is_empty() {
        vec!["X".to_string(); positions.len()]
    } else {
        atoms.to_vec()
    };

    Ok(Structure { lattice, rec_basis, volume, species, positions, selective })
}

/// Parse <varray name="selective"> — boolean triplets T/F per atom.
fn parse_selective(reader: &mut XmlReader) -> Result<Vec<[bool; 3]>> {
    let mut rows = Vec::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                let flags: Vec<bool> = text.split_whitespace()
                    .map(|s| s.eq_ignore_ascii_case("t"))
                    .collect();
                if flags.len() == 3 {
                    rows.push([flags[0], flags[1], flags[2]]);
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(rows)
}

/// Parse <crystal> block. Called after Start("crystal") event.
/// Returns (basis, rec_basis, volume).
fn parse_crystal(reader: &mut XmlReader) -> Result<([[f64; 3]; 3], [[f64; 3]; 3], f64)> {
    let mut basis: Option<[[f64; 3]; 3]> = None;
    let mut rec_basis: Option<[[f64; 3]; 3]> = None;
    let mut volume = 0.0f64;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"varray" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        match name_attr.as_str() {
                            "basis" => {
                                let rows = parse_varray_v3(reader)?;
                                if rows.len() >= 3 {
                                    basis = Some([rows[0], rows[1], rows[2]]);
                                } else {
                                    return Err(VasprunError::ShapeMismatch {
                                        expected: vec![3, 3],
                                        got: vec![rows.len()],
                                    });
                                }
                            }
                            "rec_basis" => {
                                let rows = parse_varray_v3(reader)?;
                                if rows.len() >= 3 {
                                    rec_basis = Some([rows[0], rows[1], rows[2]]);
                                } else {
                                    return Err(VasprunError::ShapeMismatch {
                                        expected: vec![3, 3],
                                        got: vec![rows.len()],
                                    });
                                }
                            }
                            _ => {
                                skip_element(reader, b"varray")?;
                            }
                        }
                    }
                    b"i" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        let text = read_text(reader)?;
                        if name_attr == "volume" {
                            volume = text.parse::<f64>().unwrap_or(0.0);
                        }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"crystal" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let basis = basis.ok_or_else(|| VasprunError::MissingElement("varray[basis]".into()))?;
    let rec_basis = rec_basis.ok_or_else(|| VasprunError::MissingElement("varray[rec_basis]".into()))?;

    Ok((basis, rec_basis, volume))
}
