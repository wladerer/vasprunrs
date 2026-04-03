use ndarray::Array2;
use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::{Dos, DosData, PartialDos};
use super::helpers::*;

/// Parse <dos> block. Called after Start("dos") event. Reads until </dos>.
pub fn parse_dos(reader: &mut XmlReader) -> Result<Dos> {
    let mut efermi = 0.0f64;
    let mut total: Option<DosData> = None;
    let mut partial: Option<PartialDos> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"i" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        let text = read_text(reader)?;
                        if name_attr == "efermi" {
                            efermi = text.parse::<f64>().unwrap_or(0.0);
                        }
                    }
                    b"total" => {
                        total = Some(parse_dos_data(reader, b"total")?);
                    }
                    b"partial" => {
                        partial = Some(parse_partial_dos(reader)?);
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"dos" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let total = total.ok_or_else(|| VasprunError::MissingElement("total inside dos".into()))?;

    Ok(Dos { efermi, total, partial })
}

/// Parse <total> or <partial> DOS array block. Called after Start("total") or Start("partial").
/// `end_tag` is the tag that terminates this block.
fn parse_dos_data(reader: &mut XmlReader, end_tag: &[u8]) -> Result<DosData> {
    let mut result: Option<DosData> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"array" => {
                result = Some(parse_dos_array(reader)?);
            }
            Event::End(ref e) if e.name().as_ref() == end_tag => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    result.ok_or_else(|| VasprunError::MissingElement("array inside dos/total".into()))
}

/// Parse <array> inside DOS total block. Called after Start("array") event.
fn parse_dos_array(reader: &mut XmlReader) -> Result<DosData> {
    // Collect spin sets from nested <set>
    let mut spin_rows: Vec<Vec<Vec<f64>>> = Vec::new(); // spin > energy_points > [energy, density, integrated]
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"set" => {
                        // Top-level set containing spin sets
                        spin_rows = parse_dos_top_set(reader)?;
                    }
                    _ => {
                        // Skip field, dimension elements
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"array" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let nspins = spin_rows.len();
    if nspins == 0 {
        return Err(VasprunError::MissingElement("spin sets in DOS".into()));
    }

    let nedos = spin_rows[0].len();
    let mut energies = Vec::with_capacity(nedos);
    let mut densities = Array2::<f64>::zeros((nspins, nedos));
    let mut integrated = Array2::<f64>::zeros((nspins, nedos));

    for (si, spin) in spin_rows.iter().enumerate() {
        for (ei, row) in spin.iter().enumerate() {
            if si == 0 {
                if let Some(&e) = row.first() {
                    energies.push(e);
                }
            }
            if row.len() >= 2 {
                densities[[si, ei]] = row[1];
            }
            if row.len() >= 3 {
                integrated[[si, ei]] = row[2];
            }
        }
    }

    Ok(DosData { energies, densities, integrated })
}

/// Parse top-level <set> containing spin sets. Returns spin > rows data.
fn parse_dos_top_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<f64>>>> {
    let mut spins = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // Spin set containing <r> rows
                let rows = parse_dos_spin_set(reader)?;
                spins.push(rows);
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

    Ok(spins)
}

/// Parse spin-level set containing <r> rows.
fn parse_dos_spin_set(reader: &mut XmlReader) -> Result<Vec<Vec<f64>>> {
    let mut rows = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"r" => {
                let text = read_text(reader)?;
                let vals = parse_floats(&text)?;
                rows.push(vals);
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

    Ok(rows)
}

/// Parse <partial> DOS block. Called after Start("partial") event. Reads until </partial>.
fn parse_partial_dos(reader: &mut XmlReader) -> Result<PartialDos> {
    let mut result: Option<PartialDos> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"array" => {
                result = Some(parse_partial_array(reader)?);
            }
            Event::End(ref e) if e.name().as_ref() == b"partial" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    result.ok_or_else(|| VasprunError::MissingElement("array inside dos/partial".into()))
}

/// Parse <array> inside partial DOS block. Called after Start("array") event.
fn parse_partial_array(reader: &mut XmlReader) -> Result<PartialDos> {
    let mut orbitals: Vec<String> = Vec::new();
    // Data: ion > spin > energy_point > orbital_values
    let mut ion_data: Vec<Vec<Vec<Vec<f64>>>> = Vec::new();
    let mut skip_first_field = true; // first field is "energy"
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"field" => {
                        let text = read_text(reader)?;
                        if skip_first_field {
                            skip_first_field = false;
                        } else {
                            orbitals.push(text);
                        }
                    }
                    b"set" => {
                        // Top-level set containing ion sets
                        ion_data = parse_partial_top_set(reader)?;
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

    let norbitals = orbitals.len();
    let nions = ion_data.len();
    let nspins = if nions > 0 { ion_data[0].len() } else { 0 };
    let nedos = if nspins > 0 { ion_data[0][0].len() } else { 0 };

    let mut data = ndarray::Array4::<f64>::zeros((nspins, nions, norbitals, nedos));

    for (ii, spins) in ion_data.iter().enumerate() {
        for (si, energy_pts) in spins.iter().enumerate() {
            for (ei, row) in energy_pts.iter().enumerate() {
                // row: [energy, orb0, orb1, ...]
                for (oi, &v) in row.iter().skip(1).enumerate().take(norbitals) {
                    data[[si, ii, oi, ei]] = v;
                }
            }
        }
    }

    Ok(PartialDos { data, orbitals })
}

/// Parse top-level <set> containing ion sets. Returns ion > spin > energy_point > values.
fn parse_partial_top_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<Vec<f64>>>>> {
    let mut ions = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // Ion set containing spin sets
                let spins = parse_partial_ion_set(reader)?;
                ions.push(spins);
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

    Ok(ions)
}

/// Parse ion-level set containing spin sets.
fn parse_partial_ion_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<f64>>>> {
    let mut spins = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // Spin set containing <r> rows
                let rows = parse_dos_spin_set(reader)?;
                spins.push(rows);
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

    Ok(spins)
}
