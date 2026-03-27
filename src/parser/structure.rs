use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::Structure;
use super::helpers::*;

pub fn parse_structure(node: Node, atoms: &[String]) -> Result<Structure> {
    let crystal = child_element(node, "crystal")
        .ok_or_else(|| VasprunError::MissingElement("crystal".into()))?;

    let basis_node = child_named(crystal, "varray", "basis")
        .ok_or_else(|| VasprunError::MissingElement("varray[basis]".into()))?;
    let lattice = parse_matrix3(basis_node)?;

    let rec_basis_node = child_named(crystal, "varray", "rec_basis")
        .ok_or_else(|| VasprunError::MissingElement("varray[rec_basis]".into()))?;
    let rec_basis = parse_matrix3(rec_basis_node)?;

    let volume = child_named(crystal, "i", "volume")
        .and_then(|n| node_text(n).parse::<f64>().ok())
        .unwrap_or(0.0);

    let pos_node = child_named(node, "varray", "positions")
        .ok_or_else(|| VasprunError::MissingElement("varray[positions]".into()))?;
    let positions = parse_varray_v3(pos_node)?;

    if !atoms.is_empty() && positions.len() != atoms.len() {
        return Err(VasprunError::ShapeMismatch {
            expected: vec![atoms.len()],
            got: vec![positions.len()],
        });
    }

    // If atoms slice is empty (shouldn't happen in practice), use placeholder.
    let species = if atoms.is_empty() {
        vec!["X".to_string(); positions.len()]
    } else {
        atoms.to_vec()
    };

    Ok(Structure { lattice, rec_basis, volume, species, positions })
}
