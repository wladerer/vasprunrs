/// Shared XML traversal and value-parsing utilities.
use roxmltree::Node;
use crate::error::{Result, VasprunError};

// ---------------------------------------------------------------------------
// Node finders
// ---------------------------------------------------------------------------

/// First child element with a given tag name.
pub fn child_element<'a>(node: Node<'a, '_>, tag: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
}

/// First child element with `name` attribute equal to `val`.
pub fn child_named<'a>(node: Node<'a, '_>, tag: &str, val: &str) -> Option<Node<'a, 'a>> {
    node.children().find(|n| {
        n.is_element()
            && n.tag_name().name() == tag
            && n.attribute("name") == Some(val)
    })
}

/// All child elements with a given tag name.
pub fn children_tagged<'a>(
    node: Node<'a, '_>,
    tag: &str,
) -> Vec<Node<'a, 'a>> {
    node.children()
        .filter(|n| n.is_element() && n.tag_name().name() == tag)
        .collect()
}

// ---------------------------------------------------------------------------
// Text extraction
// ---------------------------------------------------------------------------

pub fn node_text<'a>(node: Node<'a, '_>) -> &'a str {
    node.text().unwrap_or("").trim()
}

// ---------------------------------------------------------------------------
// Scalar parsers
// ---------------------------------------------------------------------------

pub fn parse_f64(s: &str) -> Result<f64> {
    let s = s.trim();
    // VASP writes "**********" when a value overflows its output field width.
    if s.chars().all(|c| c == '*') {
        return Ok(f64::NAN);
    }
    s.parse::<f64>().map_err(|_| VasprunError::ParseValue {
        value: s.to_string(),
        target: "f64".into(),
        reason: "invalid float literal".into(),
    })
}

pub fn parse_i64(s: &str) -> Result<i64> {
    let s = s.trim();
    s.parse::<i64>().map_err(|_| VasprunError::ParseValue {
        value: s.to_string(),
        target: "i64".into(),
        reason: "invalid integer literal".into(),
    })
}

pub fn parse_bool(s: &str) -> Result<bool> {
    match s.trim().to_ascii_uppercase().as_str() {
        "T" | "TRUE" | ".TRUE." | "YES" => Ok(true),
        "F" | "FALSE" | ".FALSE." | "NO" => Ok(false),
        other => Err(VasprunError::ParseValue {
            value: other.to_string(),
            target: "bool".into(),
            reason: "expected T/F/.TRUE./.FALSE.".into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Vector parsers
// ---------------------------------------------------------------------------

/// Parse a whitespace-separated list of f64 from a node's text.
pub fn parse_vec_f64(node: Node) -> Result<Vec<f64>> {
    node_text(node)
        .split_whitespace()
        .map(|s| parse_f64(s))
        .collect()
}

/// Parse a whitespace-separated list of i32.
pub fn parse_vec_i32(node: Node) -> Result<Vec<i32>> {
    node_text(node)
        .split_whitespace()
        .map(|s| {
            s.trim().parse::<i32>().map_err(|_| VasprunError::ParseValue {
                value: s.to_string(),
                target: "i32".into(),
                reason: "invalid integer".into(),
            })
        })
        .collect()
}

/// Parse exactly 3 f64 from a `<v>` node into a fixed array.
pub fn parse_v3(node: Node) -> Result<[f64; 3]> {
    let vals = parse_vec_f64(node)?;
    if vals.len() < 3 {
        return Err(VasprunError::ParseValue {
            value: node_text(node).to_string(),
            target: "[f64; 3]".into(),
            reason: format!("expected 3 values, got {}", vals.len()),
        });
    }
    Ok([vals[0], vals[1], vals[2]])
}

/// Parse a `<varray name="...">` of `<v>` rows into a flat Vec<Vec<f64>>.
pub fn parse_varray(node: Node) -> Result<Vec<Vec<f64>>> {
    children_tagged(node, "v")
        .iter()
        .map(|v| parse_vec_f64(*v))
        .collect()
}

/// Parse a `<varray>` of 3-vectors into Vec<[f64;3]>.
pub fn parse_varray_v3(node: Node) -> Result<Vec<[f64; 3]>> {
    children_tagged(node, "v")
        .iter()
        .map(|v| parse_v3(*v))
        .collect()
}

/// Parse 3 `<v>` rows into a 3×3 matrix.
pub fn parse_matrix3(node: Node) -> Result<[[f64; 3]; 3]> {
    let rows = parse_varray_v3(node)?;
    if rows.len() < 3 {
        return Err(VasprunError::ShapeMismatch {
            expected: vec![3, 3],
            got: vec![rows.len()],
        });
    }
    Ok([rows[0], rows[1], rows[2]])
}

/// Parse all `<r>` rows in a `<set>` node into Vec<Vec<f64>>.
pub fn parse_r_rows(node: Node) -> Result<Vec<Vec<f64>>> {
    children_tagged(node, "r")
        .iter()
        .map(|r| parse_vec_f64(*r))
        .collect()
}
