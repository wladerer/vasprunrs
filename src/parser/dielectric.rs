use ndarray::Array2;
use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::Dielectric;
use super::helpers::*;

/// Parse a <dielectricfunction> block. Called after Start("dielectricfunction") event.
/// Contains <imag> and optionally <real>, each wrapping an <array> of rows:
///   energy  xx  yy  zz  xy  yz  zx
pub fn parse_dielectric(reader: &mut XmlReader) -> Result<Dielectric> {
    let mut energies: Option<Vec<f64>> = None;
    let mut imag: Option<Array2<f64>> = None;
    let mut real: Option<Array2<f64>> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"imag" => {
                        let (en, arr) = parse_component(reader, b"imag")?;
                        energies = Some(en);
                        imag = Some(arr);
                    }
                    b"real" => {
                        let (_, arr) = parse_component(reader, b"real")?;
                        real = Some(arr);
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"dielectricfunction" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let energies = energies.ok_or_else(|| VasprunError::MissingElement("imag inside dielectricfunction".into()))?;
    let imag = imag.ok_or_else(|| VasprunError::MissingElement("imag inside dielectricfunction".into()))?;
    let real = real.unwrap_or_else(|| Array2::<f64>::zeros(imag.raw_dim()));

    Ok(Dielectric { energies, real, imag })
}

/// Parse <imag> or <real> component block. Called after Start event.
/// Returns (energies, [nfreq × 6] array).
fn parse_component(reader: &mut XmlReader, end_tag: &[u8]) -> Result<(Vec<f64>, Array2<f64>)> {
    let mut result: Option<(Vec<f64>, Array2<f64>)> = None;
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"array" => {
                result = Some(parse_dielectric_array(reader)?);
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

    result.ok_or_else(|| VasprunError::MissingElement("array inside dielectric component".into()))
}

/// Parse <array> inside dielectric component. Called after Start("array") event.
fn parse_dielectric_array(reader: &mut XmlReader) -> Result<(Vec<f64>, Array2<f64>)> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"set" => {
                        rows = parse_dielectric_set(reader)?;
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

    let nfreq = rows.len();
    let mut energies = Vec::with_capacity(nfreq);
    let mut data = Array2::<f64>::zeros((nfreq, 6));

    for (i, row) in rows.iter().enumerate() {
        if let Some(&e) = row.first() {
            energies.push(e);
        }
        for (j, &v) in row.iter().skip(1).enumerate().take(6) {
            data[[i, j]] = v;
        }
    }

    Ok((energies, data))
}

/// Parse <set> inside dielectric array. Called after Start("set") event.
fn parse_dielectric_set(reader: &mut XmlReader) -> Result<Vec<Vec<f64>>> {
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
