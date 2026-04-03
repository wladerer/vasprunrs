use ndarray::Array2;
use quick_xml::events::Event;
use crate::error::{Result, VasprunError};
use crate::types::{KpointGeneration, Kpoints};
use super::helpers::*;

/// Parse <kpoints> block. Called after Start("kpoints") event.
pub fn parse_kpoints(reader: &mut XmlReader) -> Result<Kpoints> {
    let mut generation: Option<KpointGeneration> = None;
    let mut kpointlist: Option<Vec<[f64; 3]>> = None;
    let mut weights: Option<Vec<f64>> = None;
    let mut labels_raw: Vec<String> = Vec::new();
    let mut labelindex_raw: Vec<usize> = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"generation" => {
                        let scheme = attr_str(e, b"param").unwrap_or_else(|| "unknown".into());
                        let gen = parse_generation_block(reader, scheme)?;
                        generation = Some(gen);
                    }
                    b"varray" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        match name_attr.as_str() {
                            "kpointlist" => {
                                kpointlist = Some(parse_varray_v3(reader)?);
                            }
                            "weights" => {
                                weights = Some(parse_varray_first_f64(reader)?);
                            }
                            "labels" => {
                                labels_raw = parse_varray_strings(reader)?;
                            }
                            "labelindex" => {
                                labelindex_raw = parse_varray_indices(reader)?;
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
            Event::End(ref e) if e.name().as_ref() == b"kpoints" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    let kpt_rows = kpointlist
        .ok_or_else(|| VasprunError::MissingElement("varray[kpointlist]".into()))?;
    let nkpts = kpt_rows.len();
    let mut kpointlist_arr = Array2::<f64>::zeros((nkpts, 3));
    for (i, row) in kpt_rows.iter().enumerate() {
        kpointlist_arr[[i, 0]] = row[0];
        kpointlist_arr[[i, 1]] = row[1];
        kpointlist_arr[[i, 2]] = row[2];
    }

    let weights = weights.ok_or_else(|| VasprunError::MissingElement("varray[weights]".into()))?;

    // Build labels from labelindex + labels arrays
    let labels: Vec<(usize, String)> = if !labels_raw.is_empty() && !labelindex_raw.is_empty() {
        labelindex_raw.into_iter().zip(labels_raw).collect()
    } else {
        Vec::new()
    };

    Ok(Kpoints {
        generation,
        kpointlist: kpointlist_arr,
        weights,
        labels,
    })
}

/// Parse <generation> block. Called after Start("generation") event with scheme already extracted.
fn parse_generation_block(reader: &mut XmlReader, scheme: String) -> Result<KpointGeneration> {
    let mut divisions: Option<[i32; 3]> = None;
    let mut usershift: [f64; 3] = [0.0; 3];
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                match tag.as_slice() {
                    b"v" => {
                        let name_attr = attr_str(e, b"name").unwrap_or_default();
                        let type_attr = attr_str(e, b"type").unwrap_or_default();
                        let text = read_text(reader)?;
                        match name_attr.as_str() {
                            "divisions" => {
                                if type_attr == "int" {
                                    let vals: Vec<i32> = text
                                        .split_whitespace()
                                        .filter_map(|s| s.parse().ok())
                                        .collect();
                                    if vals.len() >= 3 {
                                        divisions = Some([vals[0], vals[1], vals[2]]);
                                    }
                                }
                            }
                            "usershift" => {
                                if let Ok(v) = parse_v3(&text) {
                                    usershift = v;
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        skip_element(reader, &tag)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"generation" => break,
            Event::Eof => break,
            _ => {}
        }
    }

    Ok(KpointGeneration { scheme, divisions, usershift })
}

/// Parse a <varray> of 3-vectors. Called after Start("varray") event.
pub fn parse_varray_v3(reader: &mut XmlReader) -> Result<Vec<[f64; 3]>> {
    let mut rows = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                let v = parse_v3(&text)?;
                rows.push(v);
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
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

/// Parse a <varray> of rows, taking only the first float from each <v>.
fn parse_varray_first_f64(reader: &mut XmlReader) -> Result<Vec<f64>> {
    let mut vals = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                let floats = parse_floats(&text)?;
                if let Some(&v) = floats.first() {
                    vals.push(v);
                } else {
                    return Err(VasprunError::ParseValue {
                        value: String::new(),
                        target: "f64".into(),
                        reason: "empty weight row".into(),
                    });
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(vals)
}

/// Parse <varray> containing string label <v> rows.
fn parse_varray_strings(reader: &mut XmlReader) -> Result<Vec<String>> {
    let mut labels = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                labels.push(text);
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(labels)
}

/// Parse a `<varray>` where each `<v>` row may contain a variable number of
/// floats.  Used for magnetization (1 float per atom for collinear ISPIN=2,
/// 3 floats for non-collinear mx/my/mz).  Called after Start("varray") event.
pub fn parse_varray_floats(reader: &mut XmlReader) -> Result<Vec<Vec<f64>>> {
    let mut rows = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                rows.push(parse_floats(&text)?);
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
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

/// Parse <varray> of integer indices (labelindex), returns 0-based indices.
fn parse_varray_indices(reader: &mut XmlReader) -> Result<Vec<usize>> {
    let mut indices = Vec::new();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"v" => {
                let text = read_text(reader)?;
                let parts: Vec<i32> = text
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if let Some(&i) = parts.first() {
                    indices.push((i as usize).saturating_sub(1));
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"varray" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }

    Ok(indices)
}
