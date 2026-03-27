use ndarray::Array2;
use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::{Dos, DosData, PartialDos};
use super::helpers::*;

pub fn parse_dos(node: Node) -> Result<Dos> {
    let efermi = child_named(node, "i", "efermi")
        .and_then(|n| node_text(n).parse::<f64>().ok())
        .unwrap_or(0.0);

    let total_node = child_element(node, "total")
        .ok_or_else(|| VasprunError::MissingElement("total inside dos".into()))?;
    let total = parse_dos_data(total_node)?;

    let partial = child_element(node, "partial")
        .map(|n| parse_partial_dos(n))
        .transpose()?;

    Ok(Dos { efermi, total, partial })
}

/// Parse <total> or inner <array> blocks: spin > gridpoints rows of [energy, dos, integrated].
fn parse_dos_data(node: Node) -> Result<DosData> {
    let array = child_element(node, "array")
        .ok_or_else(|| VasprunError::MissingElement("array inside dos/total".into()))?;

    let top_set = child_element(array, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside dos array".into()))?;

    let spin_sets = children_tagged(top_set, "set");
    let nspins = spin_sets.len();
    if nspins == 0 {
        return Err(VasprunError::MissingElement("spin sets in DOS".into()));
    }

    let nedos = spin_sets[0].children()
        .filter(|n| n.is_element() && n.tag_name().name() == "r")
        .count();

    let mut energies = Vec::with_capacity(nedos);
    let mut densities  = Array2::<f64>::zeros((nspins, nedos));
    let mut integrated = Array2::<f64>::zeros((nspins, nedos));

    for (si, spin_set) in spin_sets.iter().enumerate() {
        for (ei, r) in spin_set
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "r")
            .enumerate()
        {
            let vals = parse_vec_f64(r)?;
            if si == 0 && vals.len() >= 1 {
                energies.push(vals[0]);
            }
            if vals.len() >= 2 { densities[[si, ei]]  = vals[1]; }
            if vals.len() >= 3 { integrated[[si, ei]] = vals[2]; }
        }
    }

    Ok(DosData { energies, densities, integrated })
}

/// Parse <partial> DOS: spin > ion rows of orbital values.
fn parse_partial_dos(node: Node) -> Result<PartialDos> {
    let array = child_element(node, "array")
        .ok_or_else(|| VasprunError::MissingElement("array inside dos/partial".into()))?;

    let orbitals: Vec<String> = array
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "field")
        .skip(1) // first field is "energy"
        .map(|f| node_text(f).to_string())
        .collect();
    let norbitals = orbitals.len();

    let top_set = child_element(array, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside partial dos".into()))?;

    // XML layout: outer_set > ion_set > spin_set > r
    let ion_sets = children_tagged(top_set, "set");
    let nions = ion_sets.len();

    let spin_sets_0 = ion_sets.first()
        .map(|s| children_tagged(*s, "set"))
        .unwrap_or_default();
    let nspins = spin_sets_0.len();
    let nedos = spin_sets_0.first()
        .map(|s| s.children().filter(|n| n.is_element() && n.tag_name().name() == "r").count())
        .unwrap_or(0);

    let mut data = ndarray::Array4::<f64>::zeros((nspins, nions, norbitals, nedos));

    for (ii, ion_set) in ion_sets.iter().enumerate() {
        for (si, spin_set) in children_tagged(*ion_set, "set").iter().enumerate() {
            for (ei, r) in spin_set
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "r")
                .enumerate()
            {
                let vals = parse_vec_f64(r)?;
                // cols: energy, orbital0, orbital1, ...
                for (oi, &v) in vals.iter().skip(1).enumerate().take(norbitals) {
                    data[[si, ii, oi, ei]] = v;
                }
            }
        }
    }

    Ok(PartialDos { data, orbitals })
}
