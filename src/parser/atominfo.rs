use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::{AtomInfo, AtomType};
use super::helpers::*;

pub fn parse_atominfo(node: Node) -> Result<AtomInfo> {
    // ---- per-atom element list (<array name="atoms">) ----------------------
    let atoms_array = child_named(node, "array", "atoms")
        .ok_or_else(|| VasprunError::MissingElement("array[atoms]".into()))?;
    let atoms = parse_atom_list(atoms_array)?;

    // ---- per-species metadata (<array name="atomtypes">) -------------------
    let types_array = child_named(node, "array", "atomtypes")
        .ok_or_else(|| VasprunError::MissingElement("array[atomtypes]".into()))?;
    let atom_types = parse_atom_types(types_array)?;

    Ok(AtomInfo { atoms, atom_types })
}

/// Extract element symbol for every atom from the <array name="atoms"> block.
fn parse_atom_list(node: Node) -> Result<Vec<String>> {
    let set = child_element(node, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside array[atoms]".into()))?;

    let atoms: Vec<String> = set
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "rc")
        .map(|rc| {
            // first <c> is the element symbol
            rc.children()
                .find(|n| n.is_element() && n.tag_name().name() == "c")
                .map(|c| node_text(c).to_string())
                .unwrap_or_default()
        })
        .collect();

    Ok(atoms)
}

fn parse_atom_types(node: Node) -> Result<Vec<AtomType>> {
    let set = child_element(node, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside array[atomtypes]".into()))?;

    // Determine column order from <field> elements.
    // Typical order: atomspertype, element, mass, valence, pseudopotential
    let fields: Vec<String> = node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "field")
        .map(|f| node_text(f).to_string())
        .collect();

    let types: Result<Vec<AtomType>> = set
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "rc")
        .map(|rc| {
            let cols: Vec<String> = rc
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "c")
                .map(|c| node_text(c).to_string())
                .collect();

            let get = |name: &str| -> &str {
                fields
                    .iter()
                    .position(|f| f == name)
                    .and_then(|i| cols.get(i))
                    .map(|s| s.as_str())
                    .unwrap_or("")
            };

            let count = get("atomspertype")
                .trim()
                .parse::<usize>()
                .unwrap_or(0);
            let mass = get("mass").trim().parse::<f64>().unwrap_or(0.0);
            let valence = get("valence").trim().parse::<f64>().unwrap_or(0.0);

            Ok(AtomType {
                element:         get("element").to_string(),
                count,
                mass,
                valence,
                pseudopotential: get("pseudopotential").to_string(),
            })
        })
        .collect();

    types
}
