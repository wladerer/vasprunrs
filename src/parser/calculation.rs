use roxmltree::Node;
use crate::error::{Result, VasprunError};
use crate::types::{IonicStep, ScfEnergy};
use super::helpers::*;
use super::structure::parse_structure;

/// Parse all `<calculation>` nodes into ionic steps.
///
/// VASP writes one `<calculation>` per ionic step (or a single one for
/// single-point runs). Each contains multiple `<scstep>` children followed
/// by the final `<structure>`, `<varray name="forces">`, and
/// `<varray name="stress">`.
///
/// Post-processing calculations (GW, BSE, RPA) may lack a `<structure>` block;
/// those are silently skipped.
pub fn parse_ionic_steps(calc_nodes: &[Node], atoms: &[String]) -> Result<Vec<IonicStep>> {
    calc_nodes
        .iter()
        .filter_map(|calc| {
            // Skip calculation blocks that have no structure (GW, BSE, etc.)
            if calc.children().any(|n| n.is_element() && n.tag_name().name() == "structure") {
                Some(parse_single_ionic_step(*calc, atoms))
            } else {
                None
            }
        })
        .collect()
}

fn parse_single_ionic_step(calc: Node, atoms: &[String]) -> Result<IonicStep> {
    // ---- SCF steps ---------------------------------------------------------
    let scf_steps: Vec<ScfEnergy> = calc
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "scstep")
        .map(|sc| parse_energy_block(sc))
        .collect::<Result<Vec<_>>>()?;

    // ---- Final structure ---------------------------------------------------
    let structure = calc
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "structure")
        .last()
        .map(|n| parse_structure(n, atoms))
        .transpose()?
        .ok_or_else(|| VasprunError::MissingElement("structure inside calculation".into()))?;

    // ---- Forces ------------------------------------------------------------
    let forces = calc
        .children()
        .find(|n| {
            n.is_element()
                && n.tag_name().name() == "varray"
                && n.attribute("name") == Some("forces")
        })
        .map(|n| parse_varray_v3(n))
        .transpose()?
        .unwrap_or_default();

    // ---- Stress tensor (3×3 kBar) ------------------------------------------
    let stress_rows = calc
        .children()
        .find(|n| {
            n.is_element()
                && n.tag_name().name() == "varray"
                && n.attribute("name") == Some("stress")
        })
        .map(|n| parse_varray_v3(n))
        .transpose()?
        .unwrap_or_default();

    let stress = if stress_rows.len() >= 3 {
        [stress_rows[0], stress_rows[1], stress_rows[2]]
    } else {
        [[0.0; 3]; 3]
    };

    // ---- Final energy (last top-level <energy> in this calc node) ----------
    let energy = calc
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "energy")
        .last()
        .map(|n| parse_energy_fields(n))
        .unwrap_or_default();

    Ok(IonicStep { structure, forces, stress, energy, scf_steps })
}

/// Parse `<energy>` block: extract e_fr_energy, e_wo_entrp, e_0_energy.
fn parse_energy_fields(energy: Node) -> ScfEnergy {
    let mut e = ScfEnergy::default();
    for child in energy.children().filter(|n| n.is_element() && n.tag_name().name() == "i") {
        let name = child.attribute("name").unwrap_or("");
        let val: f64 = node_text(child).parse().unwrap_or(0.0);
        match name {
            "e_fr_energy" => e.e_fr_energy = val,
            "e_wo_entrp"  => e.e_wo_entrp  = val,
            "e_0_energy"  => e.e_0_energy  = val,
            _             => {}
        }
    }
    e
}

/// An `<scstep>` may contain an `<energy>` child.
fn parse_energy_block(scstep: Node) -> Result<ScfEnergy> {
    Ok(scstep
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "energy")
        .map(parse_energy_fields)
        .unwrap_or_default())
}
