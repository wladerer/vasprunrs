use thiserror::Error;

#[derive(Debug, Error)]
pub enum VasprunError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parse error: {0}")]
    Xml(#[from] roxmltree::Error),

    #[error("Encoding error: {0}")]
    Encoding(String),

    #[error("Missing element <{0}>")]
    MissingElement(String),

    #[error("Missing attribute '{attr}' on <{element}>")]
    MissingAttribute { element: String, attr: String },

    #[error("Failed to parse value '{value}' as {target}: {reason}")]
    ParseValue { value: String, target: String, reason: String },

    #[error("Unexpected array shape: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, VasprunError>;
