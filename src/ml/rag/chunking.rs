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
mod tests {
    use super::*;

    // ====================================================================
    // DocumentLoader: plaintext
    // ====================================================================

    #[test]
    fn test_load_plaintext() {
        let doc = DocumentLoader::load_plaintext("p1", "Hello world");
        assert_eq!(doc.id, "p1");
        assert_eq!(doc.text, "Hello world");
        assert_eq!(doc.format, DocumentFormat::PlainText);
        assert!(doc.metadata.is_none());
    }

    #[test]
    fn test_load_plaintext_empty() {
        let doc = DocumentLoader::load_plaintext("p2", "");
        assert_eq!(doc.text, "");
    }

    // ====================================================================
    // DocumentLoader: markdown
    // ====================================================================

    #[test]
    fn test_load_markdown_with_h1() {
        let md = "# Title\n\nSome body text.\n\n## Section\n\nMore text.";
        let doc = DocumentLoader::load_markdown("m1", md);
        assert_eq!(doc.format, DocumentFormat::Markdown);

        let meta = doc.metadata.unwrap();
        assert_eq!(meta["title"], "Title");
        let headings = meta["headings"].as_array().unwrap();
        assert!(headings.len() >= 2);
    }

    #[test]
    fn test_load_markdown_no_h1() {
        let md = "## Section Only\n\nBody text here.";
        let doc = DocumentLoader::load_markdown("m2", md);
        let meta = doc.metadata.unwrap();
        assert!(meta["title"].is_null());
        let headings = meta["headings"].as_array().unwrap();
        assert_eq!(headings.len(), 1);
    }

    #[test]
    fn test_load_markdown_multiple_h1() {
        let md = "# First\n\nContent\n\n# Second\n\nMore content";
        let doc = DocumentLoader::load_markdown("m3", md);
        let meta = doc.metadata.unwrap();
        // Title should be the first H1
        assert_eq!(meta["title"], "First");
        let headings = meta["headings"].as_array().unwrap();
        assert_eq!(headings.len(), 2);
    }

    #[test]
    fn test_load_markdown_h3_h4() {
        let md = "### Deep heading\n\n#### Deeper heading";
        let doc = DocumentLoader::load_markdown("m4", md);
        let meta = doc.metadata.unwrap();
        let headings = meta["headings"].as_array().unwrap();
        assert_eq!(headings.len(), 2);
    }

    #[test]
    fn test_load_markdown_empty() {
        let doc = DocumentLoader::load_markdown("m5", "");
        assert_eq!(doc.text, "");
    }

    // ====================================================================
    // DocumentLoader: JSON
    // ====================================================================

    #[test]
    fn test_load_json_with_text_fields() {
        let json = r#"{"title": "Test", "body": "Content", "num": 42}"#;
        let doc = DocumentLoader::load_json("j1", json, &["title", "body"]).unwrap();

        assert_eq!(doc.format, DocumentFormat::Json);
        assert!(doc.text.contains("Test"));
        assert!(doc.text.contains("Content"));
        // num is not a requested text field
        assert!(!doc.text.contains("42"));
    }

    #[test]
    fn test_load_json_all_strings() {
        let json = r#"{"a": "alpha", "b": "beta", "n": 123}"#;
        let doc = DocumentLoader::load_json("j2", json, &[]).unwrap();
        // Should extract all string values
        assert!(doc.text.contains("alpha"));
        assert!(doc.text.contains("beta"));
    }

    #[test]
    fn test_load_json_missing_text_fields() {
        let json = r#"{"title": "Test"}"#;
        let doc = DocumentLoader::load_json("j3", json, &["nonexistent"]).unwrap();
        assert_eq!(doc.text, "");
    }

    #[test]
    fn test_load_json_non_string_values() {
        let json = r#"{"count": 42, "flag": true}"#;
        let doc = DocumentLoader::load_json("j4", json, &["count", "flag"]).unwrap();
        // Non-string values are skipped
        assert_eq!(doc.text, "");
    }

    #[test]
    fn test_load_json_invalid() {
        let result = DocumentLoader::load_json("j5", "not json", &[]);
        assert!(result.is_err());
    }

    // ====================================================================
    // RecursiveTextSplitter
    // ====================================================================

    #[test]
    fn test_split_short_text() {
        let splitter = RecursiveTextSplitter::new(1000, 0);
        let result = splitter.split("Short text.");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "Short text.");
    }

    #[test]
    fn test_split_by_paragraphs() {
        let text = "Para one content here.\n\nPara two content here.\n\nPara three content here.";
        let splitter = RecursiveTextSplitter::new(30, 0);
        let result = splitter.split(text);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_split_by_sentences() {
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let splitter = RecursiveTextSplitter::new(30, 0);
        let result = splitter.split(text);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_split_byte_positions() {
        let splitter = RecursiveTextSplitter::new(1000, 0);
        let text = "Hello world.";
        let result = splitter.split(text);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 0);
        assert_eq!(result[0].2, text.len());
    }

    #[test]
    fn test_split_with_overlap() {
        let text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8";
        let splitter = RecursiveTextSplitter::new(20, 5);
        let result = splitter.split(text);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_split_empty_text() {
        let splitter = RecursiveTextSplitter::new(100, 0);
        let result = splitter.split("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_whitespace_only() {
        let splitter = RecursiveTextSplitter::new(100, 0);
        let result = splitter.split("   \n\n  ");
        assert!(result.is_empty());
    }

    #[test]
    fn test_overlap_greater_than_chunk_size() {
        // Edge case: overlap > chunk_size should still work
        let splitter = RecursiveTextSplitter::new(10, 20);
        let text = "Hello world this is a longer text for testing.";
        let result = splitter.split(text);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_utf8_multibyte() {
        let text = "Héllo wörld. Ünïcödé text.";
        let splitter = RecursiveTextSplitter::new(1000, 0);
        let result = splitter.split(text);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, text);
    }

    #[test]
    fn test_separator_exhaustion() {
        // Text with no separators at all → returned as single chunk even if > chunk_size
        let text = "abcdefghijklmnopqrstuvwxyz";
        let splitter = RecursiveTextSplitter::new(10, 0);
        let result = splitter.split(text);
        // All separators exhausted, so it's returned as-is
        assert_eq!(result.len(), 1);
    }
}
