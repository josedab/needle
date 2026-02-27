use crate::error::{NeedleError, Result};
use serde_json;

/// Supported document input formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// Plain UTF-8 text.
    PlainText,
    /// Markdown (headers extracted as metadata).
    Markdown,
    /// JSON with configurable text-field extraction.
    Json,
}

/// A loaded document ready for pipeline ingestion.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    /// Unique identifier.
    pub id: String,
    /// Extracted text content.
    pub text: String,
    /// Source format.
    pub format: DocumentFormat,
    /// Auto-extracted metadata (headings, fields, etc.).
    pub metadata: Option<serde_json::Value>,
}

/// Extracts text from various document formats.
pub struct DocumentLoader;

impl DocumentLoader {
    /// Load plain text.
    pub fn load_plaintext(id: impl Into<String>, text: &str) -> LoadedDocument {
        LoadedDocument {
            id: id.into(),
            text: text.to_string(),
            format: DocumentFormat::PlainText,
            metadata: None,
        }
    }

    /// Load Markdown, extracting title and section headings as metadata.
    pub fn load_markdown(id: impl Into<String>, md: &str) -> LoadedDocument {
        let mut title: Option<String> = None;
        let mut headings = Vec::new();
        let mut body = Vec::new();

        for line in md.lines() {
            let t = line.trim();
            if let Some(h1) = t.strip_prefix("# ") {
                if title.is_none() {
                    title = Some(h1.trim().to_string());
                }
                headings.push(h1.trim().to_string());
                body.push(h1.trim().to_string());
            } else if let Some(h) = t.strip_prefix("## ")
                .or_else(|| t.strip_prefix("### "))
                .or_else(|| t.strip_prefix("#### "))
            {
                headings.push(h.trim().to_string());
                body.push(h.trim().to_string());
            } else {
                body.push(line.to_string());
            }
        }

        let meta = serde_json::json!({
            "format": "markdown",
            "title": title,
            "headings": headings,
        });

        LoadedDocument {
            id: id.into(),
            text: body.join("\n"),
            format: DocumentFormat::Markdown,
            metadata: Some(meta),
        }
    }

    /// Load JSON, concatenating values from `text_fields` (or all strings if empty).
    pub fn load_json(
        id: impl Into<String>,
        json_str: &str,
        text_fields: &[&str],
    ) -> Result<LoadedDocument> {
        let value: serde_json::Value =
            serde_json::from_str(json_str).map_err(NeedleError::Serialization)?;

        let mut parts = Vec::new();
        if let serde_json::Value::Object(map) = &value {
            if text_fields.is_empty() {
                for val in map.values() {
                    if let serde_json::Value::String(s) = val {
                        parts.push(s.clone());
                    }
                }
            } else {
                for field in text_fields {
                    if let Some(serde_json::Value::String(s)) = map.get(*field) {
                        parts.push(s.clone());
                    }
                }
            }
        }

        Ok(LoadedDocument {
            id: id.into(),
            text: parts.join("\n\n"),
            format: DocumentFormat::Json,
            metadata: Some(value),
        })
    }
}

/// Splits text by trying separators in order: paragraphs → lines → sentences → words.
pub struct RecursiveTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<&'static str>,
}

impl RecursiveTextSplitter {
    /// Create with defaults: paragraph → line → sentence → word.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec!["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        }
    }

    /// Split `text` into `(chunk, start_byte, end_byte)` tuples.
    pub fn split(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut results = Vec::new();
        self.split_inner(text, 0, 0, &mut results);
        results
    }

    fn split_inner(
        &self,
        text: &str,
        base: usize,
        sep_idx: usize,
        out: &mut Vec<(String, usize, usize)>,
    ) {
        if text.len() <= self.chunk_size || sep_idx >= self.separators.len() {
            if !text.trim().is_empty() {
                out.push((text.to_string(), base, base + text.len()));
            }
            return;
        }

        let sep = self.separators[sep_idx];
        let parts: Vec<&str> = text.split(sep).collect();

        if parts.len() <= 1 {
            // Separator not found — try the next finer one
            self.split_inner(text, base, sep_idx + 1, out);
            return;
        }

        let mut chunk = String::new();
        let mut chunk_start = base;
        let mut pos = base;

        for (i, part) in parts.iter().enumerate() {
            let piece = if i < parts.len() - 1 {
                format!("{part}{sep}")
            } else {
                part.to_string()
            };

            if chunk.len() + piece.len() > self.chunk_size && !chunk.is_empty() {
                if chunk.len() > self.chunk_size {
                    self.split_inner(&chunk, chunk_start, sep_idx + 1, out);
                } else {
                    out.push((chunk.clone(), chunk_start, chunk_start + chunk.len()));
                }

                // Overlap
                let keep = if self.chunk_overlap > 0 && chunk.len() > self.chunk_overlap {
                    chunk.len() - self.chunk_overlap
                } else {
                    chunk.len()
                };
                chunk = chunk[keep..].to_string();
                chunk_start = pos - chunk.len();
            }

            chunk.push_str(&piece);
            pos += piece.len();
        }

        if !chunk.trim().is_empty() {
            if chunk.len() > self.chunk_size {
                self.split_inner(&chunk, chunk_start, sep_idx + 1, out);
            } else {
                out.push((chunk.clone(), chunk_start, chunk_start + chunk.len()));
            }
        }
    }
}

#[cfg(test)]
mod tests {}
