/// Parse `<eigenvalues>` and `<projected>` blocks using quick-xml streaming.
///
/// The XML nesting is:  array > set[spin] > set[kpt] > r[band]
/// Each `<r>` contains two values: energy and occupancy.
use ndarray::{Array4, Array5};
use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::{Eigenvalues, Projected};
use super::helpers::*;

/// Parse <eigenvalues> block. Called after Start("eigenvalues") event.
pub fn parse_eigenvalues(reader: &mut XmlReader) -> Result<Eigenvalues> {
    let mut result: Option<Eigenvalues> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"array" => {
                let (data, nspins, nkpts, nbands) = parse_spin_kpt_band_array(reader)?;
                result = Some(Eigenvalues { nspins, nkpts, nbands, data });
            }
            Event::End(ref e) if e.name().as_ref() == b"eigenvalues" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    result.ok_or_else(|| VasprunError::MissingElement("array inside eigenvalues".into()))
}

/// Parse <projected> block. Called after Start("projected") event.
pub fn parse_projected(reader: &mut XmlReader) -> Result<Projected> {
    let mut result: Option<Projected> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"array" => {
                result = Some(parse_projected_array(reader)?);
            }
            Event::End(ref e) if e.name().as_ref() == b"projected" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    result.ok_or_else(|| VasprunError::MissingElement("array inside projected".into()))
}

/// Parse the <array> inside <projected>. Called after Start("array") event.
fn parse_projected_array(reader: &mut XmlReader) -> Result<Projected> {
    // Collect <field> elements for orbital labels, then find <set>
    let mut orbitals: Vec<String> = Vec::new();
    let mut spin_data: Vec<Vec<Vec<Vec<Vec<f64>>>>> = Vec::new(); // spin > kpt > band > ion > orbitals
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"field" => {
                        let text = read_text(reader)?;
                        orbitals.push(text);
                    }
                    b"set" => {
                        // Top-level set: contains spin sets
                        spin_data = parse_projected_top_set(reader)?;
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

    let norbitals = orbitals.len().max(1);
    let nspins = spin_data.len();
    if nspins == 0 {
        return Err(VasprunError::MissingElement("spin sets in projected".into()));
    }
    let nkpts = spin_data[0].len();
    let nbands = if nkpts > 0 { spin_data[0][0].len() } else { 0 };
    let nions = if nbands > 0 { spin_data[0][0][0].len() } else { 0 };

    let mut data = Array5::<f64>::zeros((nspins, nkpts, nbands, nions, norbitals));

    for (si, kpt_data) in spin_data.iter().enumerate() {
        for (ki, band_data) in kpt_data.iter().enumerate() {
            for (bi, ion_data) in band_data.iter().enumerate() {
                for (ii, orb_vals) in ion_data.iter().enumerate() {
                    for (oi, &v) in orb_vals.iter().enumerate().take(norbitals) {
                        data[[si, ki, bi, ii, oi]] = v;
                    }
                }
            }
        }
    }

    Ok(Projected { nspins, nkpts, nbands, nions, norbitals, data, orbitals })
}

/// Parse top-level <set> inside projected/array. Returns spin > kpt > band > ion > orbitals.
/// Called after Start("set") event.
fn parse_projected_top_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<Vec<Vec<f64>>>>>> {
    let mut spins = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // Spin-level set
                let kpts = parse_projected_spin_set(reader)?;
                spins.push(kpts);
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

/// Parse spin-level set. Returns kpt > band > ion > orbitals.
fn parse_projected_spin_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<Vec<f64>>>>> {
    let mut kpts = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // kpt-level set
                let bands = parse_projected_kpt_set(reader)?;
                kpts.push(bands);
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

    Ok(kpts)
}

/// Parse kpt-level set. Returns band > ion > orbitals.
fn parse_projected_kpt_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<f64>>>> {
    let mut bands = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // band-level set: contains <r> rows, one per ion
                let ions = parse_r_rows(reader)?;
                bands.push(ions);
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

    Ok(bands)
}

/// Parse a set of <r> rows. Called after Start("set") event. Returns ion > orbital values.
fn parse_r_rows(reader: &mut XmlReader) -> Result<Vec<Vec<f64>>> {
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

/// Core helper: parse spin > kpt > band(r) nesting into Array4<f64>
/// shape [nspins, nkpts, nbands, 2]. Called after Start("array") event.
pub fn parse_spin_kpt_band_array(reader: &mut XmlReader) -> Result<(Array4<f64>, usize, usize, usize)> {
    // Collect data: spin > kpt > band > [energy, occ]
    let mut spin_data: Vec<Vec<Vec<[f64; 2]>>> = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"set" => {
                        // Top-level set containing spin sets
                        spin_data = parse_eigen_top_set(reader)?;
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

    let nspins = spin_data.len();
    if nspins == 0 {
        return Err(VasprunError::MissingElement("spin sets in eigenvalues".into()));
    }

    let nkpts = spin_data[0].len();
    let nbands = if nkpts > 0 { spin_data[0][0].len() } else { 0 };

    let mut data = Array4::<f64>::zeros((nspins, nkpts, nbands, 2));

    for (si, kpt_data) in spin_data.iter().enumerate() {
        for (ki, band_data) in kpt_data.iter().enumerate() {
            for (bi, vals) in band_data.iter().enumerate() {
                data[[si, ki, bi, 0]] = vals[0];
                data[[si, ki, bi, 1]] = vals[1];
            }
        }
    }

    Ok((data, nspins, nkpts, nbands))
}

/// Parse top-level <set> inside eigenvalue array. Returns spin > kpt > band data.
fn parse_eigen_top_set(reader: &mut XmlReader) -> Result<Vec<Vec<Vec<[f64; 2]>>>> {
    let mut spins = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // Spin-level set
                let kpts = parse_eigen_spin_set(reader)?;
                spins.push(kpts);
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

/// Parse spin-level set. Returns kpt > band data.
fn parse_eigen_spin_set(reader: &mut XmlReader) -> Result<Vec<Vec<[f64; 2]>>> {
    let mut kpts = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"set" => {
                // kpt-level set: contains <r> rows, one per band
                let bands = parse_eigen_kpt_set(reader)?;
                kpts.push(bands);
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

    Ok(kpts)
}

/// Parse kpt-level set. Returns band data as Vec<[f64;2]>.
fn parse_eigen_kpt_set(reader: &mut XmlReader) -> Result<Vec<[f64; 2]>> {
    let mut bands = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"r" => {
                let text = read_text(reader)?;
                let vals = parse_floats(&text)?;
                if vals.len() < 2 {
                    return Err(VasprunError::ShapeMismatch {
                        expected: vec![2],
                        got: vec![vals.len()],
                    });
                }
                bands.push([vals[0], vals[1]]);
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

    Ok(bands)
}
