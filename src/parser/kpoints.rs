use ndarray::Array2;
use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::{KpointGeneration, Kpoints};
use super::helpers::*;

pub fn parse_kpoints(node: Node) -> Result<Kpoints> {
    // ---- generation metadata ------------------------------------------------
    let generation = child_element(node, "generation").map(|g| {
        let scheme = g.attribute("param").unwrap_or("unknown").to_string();
        let divisions = child_named(g, "v", "divisions")
            .and_then(|v| parse_vec_i32(v).ok())
            .and_then(|v| {
                if v.len() >= 3 { Some([v[0], v[1], v[2]]) } else { None }
            });
        let usershift = child_named(g, "v", "usershift")
            .and_then(|v| parse_v3(v).ok())
            .unwrap_or([0.0; 3]);

        KpointGeneration { scheme, divisions, usershift }
    });

    // ---- kpoint list --------------------------------------------------------
    let kplist_node = child_named(node, "varray", "kpointlist")
        .ok_or_else(|| VasprunError::MissingElement("varray[kpointlist]".into()))?;

    let kpt_rows = parse_varray_v3(kplist_node)?;
    let nkpts = kpt_rows.len();
    let mut kpointlist = Array2::<f64>::zeros((nkpts, 3));
    for (i, row) in kpt_rows.iter().enumerate() {
        kpointlist[[i, 0]] = row[0];
        kpointlist[[i, 1]] = row[1];
        kpointlist[[i, 2]] = row[2];
    }

    // ---- weights ------------------------------------------------------------
    let weights_node = child_named(node, "varray", "weights")
        .ok_or_else(|| VasprunError::MissingElement("varray[weights]".into()))?;

    // Weights are stored as single-element <v> rows.
    let weights: Result<Vec<f64>> = weights_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "v")
        .map(|v| {
            let vals = parse_vec_f64(v)?;
            vals.into_iter()
                .next()
                .ok_or_else(|| VasprunError::ParseValue {
                    value: String::new(),
                    target: "f64".into(),
                    reason: "empty weight row".into(),
                })
        })
        .collect();
    let weights = weights?;

    // ---- optional k-point labels (line-mode / band structure) ---------------
    let labels = parse_kpoint_labels(node);

    Ok(Kpoints { generation, kpointlist, weights, labels })
}

/// Parse `<varray name="labels">` and `<varray name="labelindex">` if present.
fn parse_kpoint_labels(node: Node) -> Vec<(usize, String)> {
    let label_node = match child_named(node, "varray", "labels") {
        Some(n) => n,
        None    => return vec![],
    };
    let idx_node = match child_named(node, "varray", "labelindex") {
        Some(n) => n,
        None    => return vec![],
    };

    let labels: Vec<String> = label_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "v")
        .map(|v| node_text(v).to_string())
        .collect();

    let indices: Vec<usize> = idx_node
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "v")
        .filter_map(|v| {
            parse_vec_i32(v)
                .ok()
                .and_then(|vals| vals.first().map(|&i| (i as usize).saturating_sub(1)))
        })
        .collect();

    indices.into_iter().zip(labels).collect()
}
