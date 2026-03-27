use ndarray::Array2;
use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::Dielectric;
use super::helpers::*;

/// Parse a `<dielectricfunction>` node.
/// Contains <imag> and optionally <real>, each wrapping an <array> of rows:
///   energy  xx  yy  zz  xy  yz  zx
pub fn parse_dielectric(node: Node) -> Result<Dielectric> {
    let imag_node = child_element(node, "imag")
        .ok_or_else(|| VasprunError::MissingElement("imag inside dielectricfunction".into()))?;
    let (energies, imag) = parse_component(imag_node)?;

    let real = child_element(node, "real")
        .map(|n| parse_component(n).map(|(_, arr)| arr))
        .transpose()?
        .unwrap_or_else(|| Array2::<f64>::zeros(imag.raw_dim()));

    Ok(Dielectric { energies, real, imag })
}

/// Returns (energies, [nfreq × 6] tensor components array).
fn parse_component(node: Node) -> Result<(Vec<f64>, Array2<f64>)> {
    let array = child_element(node, "array")
        .ok_or_else(|| VasprunError::MissingElement("array inside dielectric component".into()))?;

    let set = child_element(array, "set")
        .ok_or_else(|| VasprunError::MissingElement("set inside dielectric array".into()))?;

    let rows: Vec<Vec<f64>> = set
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "r")
        .map(|r| parse_vec_f64(r))
        .collect::<Result<_>>()?;

    let nfreq = rows.len();
    let mut energies = Vec::with_capacity(nfreq);
    let mut data = Array2::<f64>::zeros((nfreq, 6));

    for (i, row) in rows.iter().enumerate() {
        if let Some(&e) = row.first() { energies.push(e); }
        for (j, &v) in row.iter().skip(1).enumerate().take(6) {
            data[[i, j]] = v;
        }
    }

    Ok((energies, data))
}
