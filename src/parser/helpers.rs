/// Shared XML traversal and value-parsing utilities for quick-xml streaming parser.
use memchr::memmem;
use quick_xml::Reader;
use quick_xml::events::Event;

// ---------------------------------------------------------------------------
// VasprunParser — wraps Reader<&[u8]> with byte-range tracking for fast_skip
// ---------------------------------------------------------------------------

/// Streaming XML parser with SIMD-accelerated element skipping.
///
/// Wraps `Reader<&'a [u8]>` together with the original byte slice and a base
/// offset so that `absolute_pos()` always points into the original buffer.
/// This lets `fast_skip` use `memmem` to jump over large blocks (e.g.
/// `<projected>`) in O(bytes) rather than O(XML events).
pub struct VasprunParser<'a> {
    reader: Reader<&'a [u8]>,
    bytes:  &'a [u8],   // full original slice — never changes
    base:   usize,      // absolute byte offset where `reader` was created from
}

impl<'a> VasprunParser<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let reader = Self::make_reader(bytes);
        Self { reader, bytes, base: 0 }
    }

    /// Create a sub-reader starting at `bytes` with trim_text enabled.
    ///
    /// `check_end_names` is disabled because sub-readers created by `fast_skip`
    /// start mid-document and will encounter closing tags (e.g. `</calculation>`)
    /// whose corresponding opens are not visible to that reader.
    fn make_reader(bytes: &'a [u8]) -> Reader<&'a [u8]> {
        let mut r = Reader::from_reader(bytes);
        r.config_mut().trim_text(true);
        r.config_mut().check_end_names = false;
        r.config_mut().allow_unmatched_ends = true;
        r
    }

    /// Absolute byte offset of the reader's current position in `self.bytes`.
    #[inline]
    fn absolute_pos(&self) -> usize {
        self.base + self.reader.buffer_position() as usize
    }

    /// SIMD-accelerated skip to just after `</end_tag>`.
    ///
    /// Much faster than event-driven `skip_element` for large blocks because
    /// it uses `memmem::find` (SIMD byte search) rather than parsing every XML
    /// event in the skipped region.
    ///
    /// # Safety assumption
    /// `end_tag` must not appear as a closing tag of any *nested* element with
    /// the same name inside the block being skipped.  This is always true for
    /// the VASP XML tags we skip (`projected`, `eigenvalues`, `dos`,
    /// `dielectricfunction`) which never nest inside themselves.
    pub fn fast_skip(&mut self, end_tag: &[u8]) -> crate::error::Result<()> {
        let pos   = self.absolute_pos();
        let bytes: &'a [u8] = self.bytes;           // Copy &'a [u8] — no borrow of self
        let needle = build_end_tag(end_tag);
        let offset = memmem::find(&bytes[pos..], &needle)
            .ok_or_else(|| crate::error::VasprunError::MissingElement(
                format!("fast_skip: could not find </{}>",
                    std::str::from_utf8(end_tag).unwrap_or("?"))
            ))?;
        let new_base = pos + offset + needle.len();
        self.reader = Self::make_reader(&bytes[new_base..]);
        self.base   = new_base;
        Ok(())
    }

    /// Delegate `read_event_into` to the inner reader, converting errors.
    #[inline]
    pub fn read_event_into<'b>(
        &mut self,
        buf: &'b mut Vec<u8>,
    ) -> crate::error::Result<Event<'b>> {
        self.reader.read_event_into(buf).map_err(Into::into)
    }
}

/// Build a `</tag>` closing-tag needle from a raw tag name.
fn build_end_tag(tag: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(tag.len() + 3);
    v.extend_from_slice(b"</");
    v.extend_from_slice(tag);
    v.push(b'>');
    v
}

/// Every parser function takes `&mut XmlReader<'_>`.
pub type XmlReader<'a> = VasprunParser<'a>;

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/// Skip all content until the matching end tag for `tag`.
/// Call after receiving Start(tag) — reads until End(tag) at depth 0.
///
/// For large blocks prefer `reader.fast_skip(tag)` which is O(bytes) via SIMD.
pub fn skip_element(reader: &mut XmlReader, tag: &[u8]) -> crate::error::Result<()> {
    let mut buf = Vec::new();
    let mut depth = 1usize;
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Start(_) => depth += 1,
            Event::End(_) => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            Event::Eof => {
                return Err(crate::error::VasprunError::MissingElement(format!(
                    "EOF while skipping <{}>",
                    std::str::from_utf8(tag).unwrap_or("?")
                )))
            }
            _ => {}
        }
    }
    Ok(())
}

/// Read text content of current element. Call right after Start event.
/// Reads Text and then End event. Returns the trimmed text.
pub fn read_text(reader: &mut XmlReader) -> crate::error::Result<String> {
    let mut buf = Vec::new();
    let mut text = String::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Text(t) => {
                text = t
                    .unescape()
                    .map_err(|e| crate::error::VasprunError::Other(e.to_string()))?
                    .trim()
                    .to_string();
            }
            Event::End(_) => break,
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(text)
}

/// Get attribute value as String.
pub fn attr_str(e: &quick_xml::events::BytesStart, name: &[u8]) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.as_ref() == name)
        .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
}

// ---------------------------------------------------------------------------
// Scalar parsers
// ---------------------------------------------------------------------------

/// Parse space-separated f64 values from a text string.
/// VASP writes "**********" on overflow → NaN.
pub fn parse_floats(s: &str) -> crate::error::Result<Vec<f64>> {
    s.split_whitespace()
        .map(|tok| {
            if tok.chars().all(|c| c == '*') {
                return Ok(f64::NAN);
            }
            tok.parse::<f64>().map_err(|_| crate::error::VasprunError::ParseValue {
                value: tok.to_string(),
                target: "f64".into(),
                reason: "invalid float".into(),
            })
        })
        .collect()
}

pub fn parse_f64(s: &str) -> crate::error::Result<f64> {
    let s = s.trim();
    if s.chars().all(|c| c == '*') {
        return Ok(f64::NAN);
    }
    s.parse::<f64>().map_err(|_| crate::error::VasprunError::ParseValue {
        value: s.to_string(),
        target: "f64".into(),
        reason: "invalid float".into(),
    })
}

pub fn parse_i64(s: &str) -> crate::error::Result<i64> {
    s.trim().parse::<i64>().map_err(|_| crate::error::VasprunError::ParseValue {
        value: s.to_string(),
        target: "i64".into(),
        reason: "invalid integer".into(),
    })
}

pub fn parse_bool(s: &str) -> crate::error::Result<bool> {
    match s.trim().to_ascii_uppercase().as_str() {
        "T" | "TRUE" | ".TRUE." | "YES" => Ok(true),
        "F" | "FALSE" | ".FALSE." | "NO" => Ok(false),
        other => Err(crate::error::VasprunError::ParseValue {
            value: other.to_string(),
            target: "bool".into(),
            reason: "expected T/F/.TRUE./.FALSE.".into(),
        }),
    }
}

/// Parse exactly 3 f64 from a space-separated string.
pub fn parse_v3(s: &str) -> crate::error::Result<[f64; 3]> {
    let vals = parse_floats(s)?;
    if vals.len() < 3 {
        return Err(crate::error::VasprunError::ParseValue {
            value: s.to_string(),
            target: "[f64; 3]".into(),
            reason: format!("expected 3 values, got {}", vals.len()),
        });
    }
    Ok([vals[0], vals[1], vals[2]])
}
