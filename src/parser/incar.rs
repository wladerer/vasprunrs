use indexmap::IndexMap;
use quick_xml::events::Event;
use crate::error::Result;
use crate::types::{Generator, IncarValue};
use super::helpers::*;

/// Parse <generator> block. Called after Start("generator") event.
/// Reads until </generator>.
pub fn parse_generator(reader: &mut XmlReader) -> Result<Generator> {
    let mut g = Generator::default();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) => {
                let name_attr = attr_str(e, b"name").unwrap_or_default();
                let text = read_text(reader)?;
                match name_attr.as_str() {
                    "program"    => g.program    = text,
                    "version"    => g.version    = text,
                    "subversion" => g.subversion = text,
                    "platform"   => g.platform   = text,
                    "date"       => g.date       = text,
                    "time"       => g.time       = text,
                    _            => {}
                }
            }
            Event::Empty(ref e) => {
                // Self-closing <i .../> — skip
                let _ = attr_str(e, b"name");
            }
            Event::End(ref e) if e.name().as_ref() == b"generator" => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(g)
}

/// Parse <incar> block. Called after Start("incar") event.
/// Reads until </incar>.
pub fn parse_incar(reader: &mut XmlReader) -> Result<IndexMap<String, IncarValue>> {
    let mut map = IndexMap::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(ref e) if e.name().as_ref() == b"i" => {
                let name = match attr_str(e, b"name") {
                    Some(n) => n,
                    None => {
                        let _ = read_text(reader)?;
                        continue;
                    }
                };
                let type_attr = attr_str(e, b"type").unwrap_or_else(|| "float".into());
                let text = read_text(reader)?;
                let value = parse_incar_value(&text, &type_attr);
                map.insert(name, value);
            }
            Event::Empty(ref e) if e.name().as_ref() == b"i" => {
                // Self-closing <i name="X" .../> — empty value
                if let Some(name) = attr_str(e, b"name") {
                    let type_attr = attr_str(e, b"type").unwrap_or_else(|| "string".into());
                    map.insert(name, parse_incar_value("", &type_attr));
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"incar" => break,
            Event::Eof => break,
            Event::Start(ref e) => {
                let tag = e.name().as_ref().to_vec();
                skip_element(reader, &tag)?;
            }
            _ => {}
        }
    }
    Ok(map)
}

fn parse_incar_value(text: &str, type_attr: &str) -> IncarValue {
    match type_attr {
        "int" => text
            .trim()
            .parse::<i64>()
            .map(IncarValue::Int)
            .unwrap_or_else(|_| IncarValue::Str(text.to_string())),

        "logical" => match text.trim().to_uppercase().as_str() {
            "T" | ".TRUE."  | "TRUE"  => IncarValue::Bool(true),
            "F" | ".FALSE." | "FALSE" => IncarValue::Bool(false),
            _ => IncarValue::Str(text.to_string()),
        },

        "string" => IncarValue::Str(text.to_string()),

        // default: float (may be a vector of floats)
        _ => {
            let parts: Vec<f64> = text
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            match parts.len() {
                0 => IncarValue::Str(text.to_string()),
                1 => IncarValue::Float(parts[0]),
                _ => IncarValue::Vec(parts),
            }
        }
    }
}
