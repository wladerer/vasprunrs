/// Parse `<eigenvalues>` and `<projected>` blocks.
///
/// The XML nesting is:  array > set[spin] > set[kpt] > r[band]
/// Each `<r>` contains two values: energy and occupancy.
use ndarray::{Array4, Array5};
use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::{Eigenvalues, Projected};
use super::helpers::*;

pub fn parse_eigenvalues(node: Node) -> Result<Eigenvalues> {
    // <eigenvalues> contains one <array>
    let array = child_element(node, "array")
        .ok_or_else(|| VasprunError::MissingElement("array inside eigenvalues".into()))?;

    let (data, nspins, nkpts, nbands) = parse_spin_kpt_band_array(array)?;

    Ok(Eigenvalues { nspins, nkpts, nbands, data })
}

pub fn parse_projected(node: Node) -> Result<Projected> {
    // <projected> contains <array> whose innermost set has orbital labels,
    // then nested as spin > kpt > band > ion, each row = orbital values.
    let array = child_element(node, "array")
        .ok_or_else(|| VasprunError::MissingElement("array inside projected".into()))?;

    // Collect orbital field names.
    let orbitals: Vec<String> = array
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "field")
        .map(|f| node_text(f).to_string())
        .collect();
    let norbitals = orbitals.len().max(1);

    let top_set = child_element(array, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside projected/array".into()))?;

    let spin_sets = children_tagged(top_set, "set");
    let nspins = spin_sets.len();

    // Determine nkpts and nbands from first spin.
    let first_spin = spin_sets.first()
        .ok_or_else(|| VasprunError::MissingElement("spin set in projected".into()))?;
    let kpt_sets = children_tagged(*first_spin, "set");
    let nkpts = kpt_sets.len();

    let band_sets_0 = children_tagged(*kpt_sets.first()
        .ok_or_else(|| VasprunError::MissingElement("kpt set in projected".into()))?, "set");
    let nbands = band_sets_0.len();
    let nions = band_sets_0.first()
        .map(|b| children_tagged(*b, "r").len())
        .unwrap_or(0);

    let mut data = Array5::<f64>::zeros((nspins, nkpts, nbands, nions, norbitals));

    for (si, spin_set) in spin_sets.iter().enumerate() {
        for (ki, kpt_set) in children_tagged(*spin_set, "set").iter().enumerate() {
            for (bi, band_set) in children_tagged(*kpt_set, "set").iter().enumerate() {
                for (ii, r) in children_tagged(*band_set, "r").iter().enumerate() {
                    let vals = parse_vec_f64(*r)?;
                    for (oi, &v) in vals.iter().enumerate().take(norbitals) {
                        data[[si, ki, bi, ii, oi]] = v;
                    }
                }
            }
        }
    }

    Ok(Projected { nspins, nkpts, nbands, nions, norbitals, data, orbitals })
}

/// Core helper: parse spin > kpt > band(r) nesting into Array4<f64>
/// shape [nspins, nkpts, nbands, 2].
pub fn parse_spin_kpt_band_array(array: Node) -> Result<(Array4<f64>, usize, usize, usize)> {
    let top_set = child_element(array, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside eigenvalue array".into()))?;

    let spin_sets = children_tagged(top_set, "set");
    let nspins = spin_sets.len();
    if nspins == 0 {
        return Err(VasprunError::MissingElement("spin sets in eigenvalues".into()));
    }

    let first_spin = &spin_sets[0];
    let kpt_sets_0 = children_tagged(*first_spin, "set");
    let nkpts = kpt_sets_0.len();
    let nbands = kpt_sets_0.first()
        .map(|k| children_tagged(*k, "r").len())
        .unwrap_or(0);

    let mut data = Array4::<f64>::zeros((nspins, nkpts, nbands, 2));

    for (si, spin_set) in spin_sets.iter().enumerate() {
        for (ki, kpt_set) in children_tagged(*spin_set, "set").iter().enumerate() {
            for (bi, r) in children_tagged(*kpt_set, "r").iter().enumerate() {
                let vals = parse_vec_f64(*r)?;
                if vals.len() >= 2 {
                    data[[si, ki, bi, 0]] = vals[0]; // energy
                    data[[si, ki, bi, 1]] = vals[1]; // occupancy
                }
            }
        }
    }

    Ok((data, nspins, nkpts, nbands))
}
