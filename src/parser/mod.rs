mod helpers;
pub mod incar;
pub mod atominfo;
pub mod kpoints;
pub mod structure;
pub mod calculation;
pub mod eigenvalues;
pub mod dos;
pub mod dielectric;

pub use helpers::*;

use roxmltree::Document;
use crate::error::{Result, VasprunError};
use crate::types::*;

/// Parse a full vasprun.xml document into a [`Vasprun`].
pub fn parse_document(doc: &Document, parse_projected: bool) -> Result<Vasprun> {
    let root = doc.root_element(); // <modeling>

    // ---- generator ----------------------------------------------------------
    let gen_node = root
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "generator")
        .ok_or_else(|| VasprunError::MissingElement("generator".into()))?;
    let generator = incar::parse_generator(gen_node)?;

    // ---- incar --------------------------------------------------------------
    let incar_node = root
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "incar")
        .ok_or_else(|| VasprunError::MissingElement("incar".into()))?;
    let incar = incar::parse_incar(incar_node)?;

    // ---- atominfo -----------------------------------------------------------
    let atominfo_node = root
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "atominfo")
        .ok_or_else(|| VasprunError::MissingElement("atominfo".into()))?;
    let atominfo = atominfo::parse_atominfo(atominfo_node)?;

    // ---- kpoints ------------------------------------------------------------
    let kpoints_node = root
        .children()
        .find(|n| n.is_element() && n.tag_name().name() == "kpoints")
        .ok_or_else(|| VasprunError::MissingElement("kpoints".into()))?;
    let kpoints = kpoints::parse_kpoints(kpoints_node)?;

    // ---- structures ---------------------------------------------------------
    let structures: Vec<_> = root
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "structure")
        .collect();

    let initial_node = structures
        .iter()
        .find(|n| n.attribute("name") == Some("initialpos"))
        .or_else(|| structures.first())
        .copied()
        .ok_or_else(|| VasprunError::MissingElement("structure[initialpos]".into()))?;
    let initial_structure = structure::parse_structure(initial_node, &atominfo.atoms)?;

    let final_node = structures
        .iter()
        .find(|n| n.attribute("name") == Some("finalpos"))
        .or_else(|| structures.last())
        .copied()
        .ok_or_else(|| VasprunError::MissingElement("structure[finalpos]".into()))?;
    let final_structure = structure::parse_structure(final_node, &atominfo.atoms)?;

    // ---- calculation (ionic steps) -----------------------------------------
    let calc_nodes: Vec<_> = root
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "calculation")
        .collect();
    let ionic_steps = calculation::parse_ionic_steps(&calc_nodes, &atominfo.atoms)?;

    // ---- eigenvalues --------------------------------------------------------
    // Located inside the last <calculation> block.
    let eigenvalues = calc_nodes
        .last()
        .and_then(|calc| {
            calc.children()
                .find(|n| n.is_element() && n.tag_name().name() == "eigenvalues")
        })
        .map(|n| eigenvalues::parse_eigenvalues(n))
        .transpose()?;

    // ---- projected eigenvalues ----------------------------------------------
    let projected = if parse_projected {
        calc_nodes
            .last()
            .and_then(|calc| {
                calc.children()
                    .find(|n| n.is_element() && n.tag_name().name() == "projected")
            })
            .map(|n| eigenvalues::parse_projected(n))
            .transpose()?
    } else {
        None
    };

    // ---- DOS ----------------------------------------------------------------
    let dos = calc_nodes
        .last()
        .and_then(|calc| {
            // Prefer the plain <dos> block over <dos comment="wannier_interpolated"> etc.
            calc.children()
                .filter(|n| n.is_element() && n.tag_name().name() == "dos")
                .find(|n| n.attribute("comment").is_none())
                .or_else(|| calc.children().find(|n| n.is_element() && n.tag_name().name() == "dos"))
        })
        .map(|n| dos::parse_dos(n))
        .transpose()?;

    let efermi = dos.as_ref().map(|d| d.efermi);

    // ---- dielectric ---------------------------------------------------------
    let dielectric = calc_nodes
        .last()
        .and_then(|calc| {
            calc.children()
                .find(|n| n.is_element() && n.tag_name().name() == "dielectricfunction")
        })
        .map(|n| dielectric::parse_dielectric(n))
        .transpose()?;

    Ok(Vasprun {
        generator,
        incar,
        atominfo,
        kpoints,
        initial_structure,
        final_structure,
        ionic_steps,
        eigenvalues,
        projected,
        dos,
        dielectric,
        efermi,
    })
}
