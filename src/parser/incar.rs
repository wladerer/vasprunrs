use indexmap::IndexMap;
use roxmltree::Node;
use crate::error::Result;
use crate::types::{Generator, IncarValue};
use super::helpers::*;

pub fn parse_generator(node: Node) -> Result<Generator> {
    let mut g = Generator::default();
    for child in node.children().filter(|n| n.is_element()) {
        let name = child.attribute("name").unwrap_or("");
        let text = node_text(child).to_string();
        match name {
            "program"    => g.program    = text,
            "version"    => g.version    = text,
            "subversion" => g.subversion = text,
            "platform"   => g.platform   = text,
            "date"       => g.date       = text,
            "time"       => g.time       = text,
            _            => {}
        }
    }
    Ok(g)
}

pub fn parse_incar(node: Node) -> Result<IndexMap<String, IncarValue>> {
    let mut map = IndexMap::new();
    for child in node.children().filter(|n| n.is_element() && n.tag_name().name() == "i") {
        let name = match child.attribute("name") {
            Some(n) => n.to_string(),
            None    => continue,
        };
        let type_attr = child.attribute("type").unwrap_or("float");
        let text = node_text(child);

        let value = parse_incar_value(text, type_attr);
        map.insert(name, value);
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

        "logical" => {
            if text.trim().to_uppercase().starts_with('T') {
                IncarValue::Bool(true)
            } else {
                IncarValue::Bool(false)
            }
        }

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
