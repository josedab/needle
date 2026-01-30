//! Notebook Integration
//!
//! Jupyter/Colab integration utilities: magic command definitions, interactive
//! search widget HTML, and rich display formatting for search results.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::notebook::{
//!     NotebookFormatter, SearchDisplayConfig, MagicCommand,
//! };
//!
//! let formatter = NotebookFormatter::new(SearchDisplayConfig::default());
//!
//! // Format search results as HTML for Jupyter
//! let html = formatter.format_results(&[
//!     ("doc1", 0.05, Some("Rust is great")),
//!     ("doc2", 0.12, Some("Python rocks")),
//! ]);
//! assert!(html.contains("<table"));
//!
//! // Get magic command definitions for Python extension
//! let magics = NotebookFormatter::magic_commands();
//! assert!(magics.iter().any(|m| m.name == "needle_search"));
//! ```

use serde::{Deserialize, Serialize};

/// Display configuration for notebook output.
#[derive(Debug, Clone)]
pub struct SearchDisplayConfig {
    pub max_text_preview: usize,
    pub show_distance: bool,
    pub show_score: bool,
    pub show_metadata: bool,
    pub color_scheme: ColorScheme,
}

impl Default for SearchDisplayConfig {
    fn default() -> Self {
        Self {
            max_text_preview: 200,
            show_distance: true,
            show_score: true,
            show_metadata: false,
            color_scheme: ColorScheme::Light,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme { Light, Dark }

/// A Jupyter magic command definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagicCommand {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub python_code: String,
}

/// Explorer widget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerWidget {
    pub html: String,
    pub javascript: String,
    pub css: String,
}

/// Notebook output formatter.
pub struct NotebookFormatter {
    config: SearchDisplayConfig,
}

impl NotebookFormatter {
    pub fn new(config: SearchDisplayConfig) -> Self { Self { config } }

    /// Format search results as an HTML table for Jupyter display.
    pub fn format_results(&self, results: &[(&str, f32, Option<&str>)]) -> String {
        let bg = if self.config.color_scheme == ColorScheme::Dark {
            "#1e1e1e"
        } else {
            "#ffffff"
        };
        let fg = if self.config.color_scheme == ColorScheme::Dark { "#e0e0e0" } else { "#333" };

        let mut html = format!(
            "<table style='border-collapse:collapse;width:100%;background:{bg};color:{fg};font-family:monospace'>\n<tr style='border-bottom:2px solid #ccc'>\n  <th style='padding:8px;text-align:left'>Rank</th>\n  <th style='padding:8px;text-align:left'>ID</th>\n"
        );
        if self.config.show_distance {
            html.push_str("  <th style='padding:8px;text-align:left'>Distance</th>\n");
        }
        if self.config.show_score {
            html.push_str("  <th style='padding:8px;text-align:left'>Score</th>\n");
        }
        html.push_str("  <th style='padding:8px;text-align:left'>Text</th>\n</tr>\n");

        for (i, (id, distance, text)) in results.iter().enumerate() {
            let score = 1.0 / (1.0 + distance);
            let preview = text.map(|t| {
                if t.len() > self.config.max_text_preview {
                    format!("{}…", &t[..self.config.max_text_preview])
                } else {
                    t.to_string()
                }
            }).unwrap_or_default();

            let score_bar = "█".repeat((score * 10.0) as usize);

            html.push_str(&format!("<tr style='border-bottom:1px solid #eee'>\n  <td style='padding:6px'>{}</td>\n  <td style='padding:6px'><code>{}</code></td>\n", i + 1, id));
            if self.config.show_distance {
                html.push_str(&format!("  <td style='padding:6px'>{:.4}</td>\n", distance));
            }
            if self.config.show_score {
                html.push_str(&format!("  <td style='padding:6px'>{:.2} {}</td>\n", score, score_bar));
            }
            html.push_str(&format!("  <td style='padding:6px'>{}</td>\n</tr>\n", preview));
        }

        html.push_str("</table>");
        html
    }

    /// Format a single result as a rich HTML card.
    pub fn format_result_card(&self, id: &str, distance: f32, text: Option<&str>) -> String {
        let score = 1.0 / (1.0 + distance);
        format!(
            "<div style='border:1px solid #ccc;border-radius:8px;padding:12px;margin:8px 0;font-family:sans-serif'>\
             <div style='font-weight:bold;color:#2563eb'>{id}</div>\
             <div style='color:#666;font-size:0.85em'>Distance: {distance:.4} | Score: {score:.2}</div>\
             <div style='margin-top:8px'>{}</div></div>",
            text.unwrap_or("(no text)")
        )
    }

    /// Generate magic command definitions for the Jupyter extension.
    pub fn magic_commands() -> Vec<MagicCommand> {
        vec![
            MagicCommand {
                name: "needle_search".into(),
                description: "Search a Needle collection with natural language".into(),
                usage: "%needle_search <collection> <query> [k=10]".into(),
                python_code: r#"
@register_line_magic
def needle_search(line):
    parts = line.split(maxsplit=2)
    collection, query = parts[0], parts[1] if len(parts) > 1 else ""
    k = 10
    import needle
    results = needle.search(collection, query, k=k)
    from IPython.display import HTML
    return HTML(needle.format_results_html(results))
"#.into(),
            },
            MagicCommand {
                name: "needle_explore".into(),
                description: "Open interactive collection explorer widget".into(),
                usage: "%needle_explore <collection>".into(),
                python_code: r#"
@register_line_magic
def needle_explore(line):
    collection = line.strip()
    import needle
    from IPython.display import HTML
    return HTML(needle.explorer_widget(collection))
"#.into(),
            },
            MagicCommand {
                name: "needle_info".into(),
                description: "Show collection information".into(),
                usage: "%needle_info <collection>".into(),
                python_code: r#"
@register_line_magic
def needle_info(line):
    collection = line.strip()
    import needle
    info = needle.collection_info(collection)
    print(f"Collection: {info['name']}")
    print(f"Vectors: {info['count']}")
    print(f"Dimensions: {info['dimensions']}")
"#.into(),
            },
        ]
    }

    /// Generate the explorer widget HTML/JS.
    pub fn explorer_widget(collection: &str, dimensions: usize) -> ExplorerWidget {
        ExplorerWidget {
            html: format!(
                "<div id='needle-explorer' data-collection='{collection}' data-dims='{dimensions}'>\
                 <h3>🔍 Needle Explorer: {collection}</h3>\
                 <input type='text' id='needle-query' placeholder='Search...' style='width:100%;padding:8px;margin:8px 0'>\
                 <div id='needle-results'></div>\
                 <canvas id='needle-viz' width='600' height='400' style='border:1px solid #ccc'></canvas>\
                 </div>"
            ),
            javascript: format!(
                "document.getElementById('needle-query').addEventListener('input', async (e) => {{\n\
                 const resp = await fetch('/collections/{collection}/search', {{\n\
                 method: 'POST', headers: {{'Content-Type': 'application/json'}},\n\
                 body: JSON.stringify({{query: e.target.value, k: 10}})\n\
                 }});\n\
                 const results = await resp.json();\n\
                 document.getElementById('needle-results').innerHTML = results.map(r => \n\
                 `<div>${{r.id}}: ${{r.distance.toFixed(4)}}</div>`).join('');\n\
                 }});"
            ),
            css: "
                #needle-explorer { font-family: sans-serif; padding: 16px; }
                #needle-explorer input { border: 1px solid #ccc; border-radius: 4px; }
                #needle-explorer h3 { margin: 0 0 12px; }
            ".into(),
        }
    }

    /// Generate the Python extension loader code.
    pub fn python_extension_code() -> String {
        r#"""Needle Jupyter Extension. Load with: %load_ext needle"""

def load_ipython_extension(ipython):
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def needle_search(line):
        """Search a Needle collection: %needle_search <collection> <query>"""
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: %needle_search <collection> <query>")
            return
        collection, query = parts[0], parts[1]
        try:
            import needle
            results = needle.search(collection, query, k=10)
            from IPython.display import HTML, display
            html = "<table><tr><th>ID</th><th>Distance</th></tr>"
            for r in results:
                html += f"<tr><td>{r.id}</td><td>{r.distance:.4f}</td></tr>"
            html += "</table>"
            display(HTML(html))
        except Exception as e:
            print(f"Error: {e}")

    @register_line_magic
    def needle_info(line):
        """Show collection info: %needle_info <collection>"""
        try:
            import needle
            info = needle.collection_info(line.strip())
            print(f"Collection: {info['name']}, Vectors: {info['count']}, Dims: {info['dimensions']}")
        except Exception as e:
            print(f"Error: {e}")
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_results_html() {
        let formatter = NotebookFormatter::new(SearchDisplayConfig::default());
        let html = formatter.format_results(&[
            ("doc1", 0.05, Some("Rust is a systems language")),
            ("doc2", 0.15, Some("Python is great")),
        ]);
        assert!(html.contains("<table"));
        assert!(html.contains("doc1"));
        assert!(html.contains("doc2"));
        assert!(html.contains("0.05"));
    }

    #[test]
    fn test_format_result_card() {
        let formatter = NotebookFormatter::new(SearchDisplayConfig::default());
        let card = formatter.format_result_card("doc1", 0.1, Some("Hello world"));
        assert!(card.contains("doc1"));
        assert!(card.contains("Hello world"));
        assert!(card.contains("Score:"));
    }

    #[test]
    fn test_dark_mode() {
        let formatter = NotebookFormatter::new(SearchDisplayConfig {
            color_scheme: ColorScheme::Dark,
            ..Default::default()
        });
        let html = formatter.format_results(&[("d1", 0.1, None)]);
        assert!(html.contains("#1e1e1e"));
    }

    #[test]
    fn test_magic_commands() {
        let magics = NotebookFormatter::magic_commands();
        assert_eq!(magics.len(), 3);
        assert!(magics.iter().any(|m| m.name == "needle_search"));
        assert!(magics.iter().any(|m| m.name == "needle_explore"));
        assert!(magics.iter().any(|m| m.name == "needle_info"));
    }

    #[test]
    fn test_explorer_widget() {
        let widget = NotebookFormatter::explorer_widget("docs", 384);
        assert!(widget.html.contains("needle-explorer"));
        assert!(widget.html.contains("docs"));
        assert!(widget.javascript.contains("fetch"));
    }

    #[test]
    fn test_python_extension_code() {
        let code = NotebookFormatter::python_extension_code();
        assert!(code.contains("load_ipython_extension"));
        assert!(code.contains("needle_search"));
        assert!(code.contains("register_line_magic"));
    }

    #[test]
    fn test_text_truncation() {
        let formatter = NotebookFormatter::new(SearchDisplayConfig {
            max_text_preview: 10,
            ..Default::default()
        });
        let html = formatter.format_results(&[("d1", 0.1, Some("This is a very long text that should be truncated"))]);
        assert!(html.contains("…"));
    }
}
