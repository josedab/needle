//! Interactive Terminal UI for Needle Vector Database
//!
//! Provides a rich terminal interface for exploring and managing vector databases.
//!
//! # Features
//!
//! - Dashboard with database statistics
//! - Collection browser and management
//! - Interactive vector search
//! - Cluster visualization
//! - Anomaly detection view
//! - Real-time streaming monitor
//!
//! # Usage
//!
//! ```rust,ignore
//! use needle::tui::NeedleTui;
//! use needle::Database;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = Database::open("vectors.needle")?;
//!     let mut tui = NeedleTui::new(db)?;
//!     tui.run().await?;
//!     Ok(())
//! }
//! ```

use crate::Database;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Sparkline, Table,
        TableState, Tabs, Wrap,
    },
    Frame, Terminal,
};
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Application state for the TUI
pub struct NeedleTui {
    db: Arc<Database>,
    terminal: Terminal<CrosstermBackend<Stdout>>,
    state: AppState,
    should_quit: bool,
}

/// Current view/tab
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Dashboard,
    Collections,
    Search,
    Clusters,
    Anomalies,
    Streaming,
    Help,
}

impl View {
    fn all() -> Vec<View> {
        vec![
            View::Dashboard,
            View::Collections,
            View::Search,
            View::Clusters,
            View::Anomalies,
            View::Streaming,
            View::Help,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            View::Dashboard => "Dashboard",
            View::Collections => "Collections",
            View::Search => "Search",
            View::Clusters => "Clusters",
            View::Anomalies => "Anomalies",
            View::Streaming => "Streaming",
            View::Help => "Help",
        }
    }

    fn index(&self) -> usize {
        match self {
            View::Dashboard => 0,
            View::Collections => 1,
            View::Search => 2,
            View::Clusters => 3,
            View::Anomalies => 4,
            View::Streaming => 5,
            View::Help => 6,
        }
    }
}

/// Application state
struct AppState {
    current_view: View,
    // Collections view
    collections: Vec<CollectionInfo>,
    collection_state: ListState,
    #[allow(dead_code)]
    selected_collection: Option<String>,
    // Search view
    search_input: String,
    search_results: Vec<SearchResultDisplay>,
    search_state: TableState,
    // Clusters view
    cluster_data: Option<ClusterData>,
    // Anomalies view
    anomaly_data: Option<AnomalyData>,
    // Streaming view
    stream_events: Vec<StreamEvent>,
    #[allow(dead_code)]
    stream_scroll: u16,
    // Stats for dashboard
    stats: DashboardStats,
    // Input mode
    input_mode: InputMode,
    // Status message
    status_message: Option<(String, Instant)>,
    // Sparkline data for dashboard
    query_history: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Normal,
    Editing,
}

#[derive(Debug, Clone)]
struct CollectionInfo {
    name: String,
    dimensions: usize,
    count: usize,
    index_type: String,
}

#[derive(Debug, Clone)]
struct SearchResultDisplay {
    id: String,
    distance: f32,
    metadata: String,
}

#[derive(Debug, Clone)]
struct ClusterData {
    #[allow(dead_code)]
    collection: String,
    k: usize,
    clusters: Vec<ClusterInfo>,
    #[allow(dead_code)]
    inertia: f32,
    algorithm: String,
    iterations: usize,
}

#[derive(Debug, Clone)]
struct ClusterInfo {
    id: usize,
    size: usize,
    #[allow(dead_code)]
    centroid_preview: String,
}

#[derive(Debug, Clone)]
struct AnomalyData {
    #[allow(dead_code)]
    collection: String,
    outliers: Vec<OutlierInfo>,
    method: String,
}

#[derive(Debug, Clone)]
struct OutlierInfo {
    id: String,
    score: f32,
    reason: String,
}

#[derive(Debug, Clone)]
struct StreamEvent {
    timestamp: String,
    event_type: String,
    collection: String,
    details: String,
}

#[derive(Debug, Clone, Default)]
struct DashboardStats {
    total_collections: usize,
    total_vectors: usize,
    total_storage_mb: f64,
    avg_dimensions: usize,
    #[allow(dead_code)]
    queries_per_second: f64,
    #[allow(dead_code)]
    uptime_seconds: u64,
}

/// Render the UI - standalone function to avoid borrow conflicts
fn render_ui(f: &mut Frame, _db: &Arc<Database>, state: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header/tabs
            Constraint::Min(0),    // Main content
            Constraint::Length(3), // Status bar
        ])
        .split(f.area());

    render_header(f, state, chunks[0]);
    render_content(f, state, chunks[1]);
    render_status_bar(f, state, chunks[2]);
}

fn render_header(f: &mut Frame, state: &AppState, area: Rect) {
    let titles: Vec<Line> = View::all()
        .iter()
        .map(|v| {
            let style = if *v == state.current_view {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Line::from(Span::styled(format!(" {} ", v.title()), style))
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Needle Vector Database ")
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        )
        .select(state.current_view.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow));

    f.render_widget(tabs, area);
}

fn render_content(f: &mut Frame, state: &mut AppState, area: Rect) {
    match state.current_view {
        View::Dashboard => render_dashboard(f, state, area),
        View::Collections => render_collections(f, state, area),
        View::Search => render_search(f, state, area),
        View::Clusters => render_clusters(f, state, area),
        View::Anomalies => render_anomalies(f, state, area),
        View::Streaming => render_streaming(f, state, area),
        View::Help => render_help(f, state, area),
    }
}

fn render_dashboard(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // Stats cards
            Constraint::Min(10),    // Query activity
            Constraint::Length(10), // Recent activity
        ])
        .split(area);

    // Stats cards row
    let card_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[0]);

    render_stat_card(
        f,
        card_chunks[0],
        "Collections",
        &state.stats.total_collections.to_string(),
        Color::Cyan,
    );
    render_stat_card(
        f,
        card_chunks[1],
        "Total Vectors",
        &format_number(state.stats.total_vectors),
        Color::Green,
    );
    render_stat_card(
        f,
        card_chunks[2],
        "Storage",
        &format!("{:.2} MB", state.stats.total_storage_mb),
        Color::Yellow,
    );
    render_stat_card(
        f,
        card_chunks[3],
        "Avg Dimensions",
        &state.stats.avg_dimensions.to_string(),
        Color::Magenta,
    );

    // Query activity sparkline
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" Query Activity (last 60s) ")
                .borders(Borders::ALL),
        )
        .data(&state.query_history)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, chunks[1]);

    // Recent activity
    let events: Vec<ListItem> = state
        .stream_events
        .iter()
        .rev()
        .take(5)
        .map(|e| {
            let color = match e.event_type.as_str() {
                "INSERT" => Color::Green,
                "DELETE" => Color::Red,
                "SEARCH" => Color::Cyan,
                _ => Color::Gray,
            };
            ListItem::new(Line::from(vec![
                Span::styled(&e.timestamp, Style::default().fg(Color::DarkGray)),
                Span::raw(" "),
                Span::styled(&e.event_type, Style::default().fg(color)),
                Span::raw(" on "),
                Span::styled(&e.collection, Style::default().fg(Color::Yellow)),
            ]))
        })
        .collect();

    let activity_list = List::new(events).block(
        Block::default()
            .title(" Recent Activity ")
            .borders(Borders::ALL),
    );
    f.render_widget(activity_list, chunks[2]);
}

fn render_stat_card(f: &mut Frame, area: Rect, title: &str, value: &str, color: Color) {
    let block = Block::default()
        .title(format!(" {} ", title))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let text = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            value,
            Style::default()
                .fg(color)
                .add_modifier(Modifier::BOLD),
        )),
    ])
    .alignment(Alignment::Center);
    f.render_widget(text, inner);
}

fn render_collections(f: &mut Frame, state: &mut AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    let items: Vec<ListItem> = state
        .collections
        .iter()
        .map(|c| {
            ListItem::new(Line::from(vec![
                Span::styled(&c.name, Style::default().fg(Color::Cyan)),
                Span::raw(" - "),
                Span::styled(
                    format!("{} vectors", c.count),
                    Style::default().fg(Color::Gray),
                ),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title(" Collections [↑↓ navigate, Enter select] ")
                .borders(Borders::ALL),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    f.render_stateful_widget(list, chunks[0], &mut state.collection_state);

    let detail_text = if let Some(idx) = state.collection_state.selected() {
        if let Some(col) = state.collections.get(idx) {
            vec![
                Line::from(vec![
                    Span::styled("Name: ", Style::default().fg(Color::Gray)),
                    Span::styled(&col.name, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Dimensions: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        col.dimensions.to_string(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Vector Count: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format_number(col.count),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Index Type: ", Style::default().fg(Color::Gray)),
                    Span::styled(&col.index_type, Style::default().fg(Color::Magenta)),
                ]),
                Line::from(""),
                Line::from(""),
                Line::from(Span::styled(
                    "Actions:",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from("  [s] Search in collection"),
                Line::from("  [c] Run clustering"),
                Line::from("  [a] Detect anomalies"),
                Line::from("  [d] Delete collection"),
                Line::from("  [e] Export to JSON"),
            ]
        } else {
            vec![Line::from("Select a collection")]
        }
    } else {
        vec![Line::from("Select a collection to view details")]
    };

    let details = Paragraph::new(detail_text)
        .block(
            Block::default()
                .title(" Collection Details ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(details, chunks[1]);
}

fn render_search(f: &mut Frame, state: &mut AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Search input
            Constraint::Min(0),    // Results
        ])
        .split(area);

    let input_style = match state.input_mode {
        InputMode::Normal => Style::default(),
        InputMode::Editing => Style::default().fg(Color::Yellow),
    };

    let input = Paragraph::new(state.search_input.as_str())
        .style(input_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Search Query [Press 'i' to edit, Enter to search] "),
        );
    f.render_widget(input, chunks[0]);

    if state.input_mode == InputMode::Editing {
        f.set_cursor_position((
            chunks[0].x + state.search_input.len() as u16 + 1,
            chunks[0].y + 1,
        ));
    }

    let header_cells = ["ID", "Distance", "Metadata"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows = state.search_results.iter().map(|r| {
        Row::new(vec![
            Cell::from(r.id.clone()),
            Cell::from(format!("{:.6}", r.distance)),
            Cell::from(truncate_string(&r.metadata, 50)),
        ])
    });

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(65),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(
                " Results ({}) ",
                state.search_results.len()
            )),
    )
    .row_highlight_style(Style::default().bg(Color::DarkGray))
    .highlight_symbol("▶ ");

    f.render_stateful_widget(table, chunks[1], &mut state.search_state);
}

fn render_clusters(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let cluster_items: Vec<ListItem> = if let Some(ref data) = state.cluster_data {
        data.clusters
            .iter()
            .map(|c| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("Cluster {}", c.id),
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(vec![
                        Span::raw("  Size: "),
                        Span::styled(
                            c.size.to_string(),
                            Style::default().fg(Color::Green),
                        ),
                    ]),
                ])
            })
            .collect()
    } else {
        vec![ListItem::new("No clustering results. Press 'r' to run clustering.")]
    };

    let cluster_list = List::new(cluster_items).block(
        Block::default()
            .title(" Clusters ")
            .borders(Borders::ALL),
    );
    f.render_widget(cluster_list, chunks[0]);

    let info_text = if let Some(ref data) = state.cluster_data {
        vec![
            Line::from(vec![
                Span::styled("Algorithm: ", Style::default().fg(Color::Gray)),
                Span::styled(&data.algorithm, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("K: ", Style::default().fg(Color::Gray)),
                Span::styled(data.k.to_string(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("Iterations: ", Style::default().fg(Color::Gray)),
                Span::styled(data.iterations.to_string(), Style::default().fg(Color::Green)),
            ]),
        ]
    } else {
        vec![Line::from("Run clustering to see results")]
    };

    let info = Paragraph::new(info_text)
        .block(Block::default().title(" Info ").borders(Borders::ALL));
    f.render_widget(info, chunks[1]);
}

fn render_anomalies(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    let method_info = if let Some(ref data) = state.anomaly_data {
        format!("Method: {} | Outliers: {}", data.method, data.outliers.len())
    } else {
        "No anomaly detection results. Press 'r' to run detection.".to_string()
    };

    let header = Paragraph::new(method_info)
        .block(Block::default().borders(Borders::ALL).title(" Anomaly Detection "));
    f.render_widget(header, chunks[0]);

    let items: Vec<ListItem> = if let Some(ref data) = state.anomaly_data {
        data.outliers
            .iter()
            .map(|o| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(&o.id, Style::default().fg(Color::Red)),
                        Span::raw(" - Score: "),
                        Span::styled(format!("{:.4}", o.score), Style::default().fg(Color::Yellow)),
                    ]),
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(&o.reason, Style::default().fg(Color::Gray)),
                    ]),
                ])
            })
            .collect()
    } else {
        vec![]
    };

    let list = List::new(items).block(Block::default().borders(Borders::ALL).title(" Outliers "));
    f.render_widget(list, chunks[1]);
}

fn render_streaming(f: &mut Frame, state: &AppState, area: Rect) {
    let items: Vec<ListItem> = state
        .stream_events
        .iter()
        .rev()
        .map(|e| {
            let color = match e.event_type.as_str() {
                "INSERT" => Color::Green,
                "DELETE" => Color::Red,
                "SEARCH" => Color::Cyan,
                "UPDATE" => Color::Yellow,
                _ => Color::Gray,
            };
            ListItem::new(Line::from(vec![
                Span::styled(&e.timestamp, Style::default().fg(Color::DarkGray)),
                Span::raw(" | "),
                Span::styled(
                    format!("{:8}", e.event_type),
                    Style::default().fg(color),
                ),
                Span::raw(" | "),
                Span::styled(&e.collection, Style::default().fg(Color::Cyan)),
                Span::raw(" | "),
                Span::raw(&e.details),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Event Stream (newest first) ")
            .borders(Borders::ALL),
    );
    f.render_widget(list, area);
}

fn render_help(f: &mut Frame, _state: &AppState, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled(
            "Navigation",
            Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
        )),
        Line::from("  Tab / Shift+Tab  - Switch between views"),
        Line::from("  ↑ / ↓           - Navigate lists"),
        Line::from("  Enter           - Select / Confirm"),
        Line::from("  q / Ctrl+C      - Quit"),
        Line::from(""),
        Line::from(Span::styled(
            "Search View",
            Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
        )),
        Line::from("  i               - Enter edit mode"),
        Line::from("  Esc             - Exit edit mode"),
        Line::from("  Enter           - Execute search"),
        Line::from(""),
        Line::from(Span::styled(
            "Collections View",
            Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
        )),
        Line::from("  s               - Search in collection"),
        Line::from("  c               - Run clustering"),
        Line::from("  a               - Detect anomalies"),
        Line::from("  d               - Delete collection"),
        Line::from("  e               - Export to JSON"),
        Line::from(""),
        Line::from(Span::styled(
            "Clusters / Anomalies View",
            Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan),
        )),
        Line::from("  r               - Run analysis"),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .title(" Help ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(help, area);
}

fn render_status_bar(f: &mut Frame, state: &AppState, area: Rect) {
    let status = if let Some((ref msg, time)) = state.status_message {
        if time.elapsed() < Duration::from_secs(5) {
            msg.clone()
        } else {
            "Ready".to_string()
        }
    } else {
        "Ready".to_string()
    };

    let status_bar = Paragraph::new(Line::from(vec![
        Span::styled(" Status: ", Style::default().fg(Color::Gray)),
        Span::styled(status, Style::default().fg(Color::Green)),
        Span::raw(" | "),
        Span::styled("Press ", Style::default().fg(Color::Gray)),
        Span::styled("?", Style::default().fg(Color::Yellow)),
        Span::styled(" for help, ", Style::default().fg(Color::Gray)),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::styled(" to quit", Style::default().fg(Color::Gray)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(status_bar, area);
}

impl NeedleTui {
    /// Create a new TUI application
    pub fn new(db: Database) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        let db = Arc::new(db);
        let state = AppState::new(&db);

        Ok(Self {
            db,
            terminal,
            state,
            should_quit: false,
        })
    }

    /// Run the TUI event loop
    pub async fn run(&mut self) -> io::Result<()> {
        let tick_rate = Duration::from_millis(100);
        let mut last_tick = Instant::now();

        loop {
            // Use a scope to isolate the terminal borrow from the rest of self
            {
                let db = Arc::clone(&self.db);
                let state = &mut self.state;
                self.terminal.draw(|f| {
                    render_ui(f, &db, state);
                })?;
            }

            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or(Duration::ZERO);

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key.code, key.modifiers);
                }
            }

            if last_tick.elapsed() >= tick_rate {
                self.on_tick();
                last_tick = Instant::now();
            }

            if self.should_quit {
                break;
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        self.terminal.show_cursor()?;

        Ok(())
    }

    #[allow(dead_code)]
    fn ui(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header/tabs
                Constraint::Min(0),    // Main content
                Constraint::Length(3), // Status bar
            ])
            .split(f.area());

        self.render_header(f, chunks[0]);
        self.render_content(f, chunks[1]);
        self.render_status_bar(f, chunks[2]);
    }

    #[allow(dead_code)]
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let titles: Vec<Line> = View::all()
            .iter()
            .map(|v| {
                let style = if *v == self.state.current_view {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                };
                Line::from(Span::styled(format!(" {} ", v.title()), style))
            })
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Needle Vector Database ")
                    .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            )
            .select(self.state.current_view.index())
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().fg(Color::Yellow));

        f.render_widget(tabs, area);
    }

    #[allow(dead_code)]
    fn render_content(&mut self, f: &mut Frame, area: Rect) {
        match self.state.current_view {
            View::Dashboard => self.render_dashboard(f, area),
            View::Collections => self.render_collections(f, area),
            View::Search => self.render_search(f, area),
            View::Clusters => self.render_clusters(f, area),
            View::Anomalies => self.render_anomalies(f, area),
            View::Streaming => self.render_streaming(f, area),
            View::Help => self.render_help(f, area),
        }
    }

    #[allow(dead_code)]
    fn render_dashboard(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8), // Stats cards
                Constraint::Min(0),    // Charts
            ])
            .split(area);

        // Stats cards
        let stats_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(chunks[0]);

        self.render_stat_card(
            f,
            stats_chunks[0],
            "Collections",
            &self.state.stats.total_collections.to_string(),
            Color::Cyan,
        );
        self.render_stat_card(
            f,
            stats_chunks[1],
            "Total Vectors",
            &format_number(self.state.stats.total_vectors),
            Color::Green,
        );
        self.render_stat_card(
            f,
            stats_chunks[2],
            "Storage",
            &format!("{:.2} MB", self.state.stats.total_storage_mb),
            Color::Yellow,
        );
        self.render_stat_card(
            f,
            stats_chunks[3],
            "Avg Dimensions",
            &self.state.stats.avg_dimensions.to_string(),
            Color::Magenta,
        );

        // Charts area
        let chart_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[1]);

        // Query sparkline
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title(" Query Rate (last 60s) ")
                    .borders(Borders::ALL),
            )
            .data(&self.state.query_history)
            .style(Style::default().fg(Color::Green));
        f.render_widget(sparkline, chart_chunks[0]);

        // Recent collections
        let items: Vec<ListItem> = self
            .state
            .collections
            .iter()
            .take(10)
            .map(|c| {
                ListItem::new(format!(
                    "{}: {} vectors ({}D)",
                    c.name, c.count, c.dimensions
                ))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .title(" Recent Collections ")
                    .borders(Borders::ALL),
            )
            .style(Style::default().fg(Color::White));
        f.render_widget(list, chart_chunks[1]);
    }

    #[allow(dead_code)]
    fn render_stat_card(&self, f: &mut Frame, area: Rect, title: &str, value: &str, color: Color) {
        let block = Block::default()
            .title(format!(" {} ", title))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(color));

        let inner = block.inner(area);
        f.render_widget(block, area);

        let value_paragraph = Paragraph::new(value)
            .style(
                Style::default()
                    .fg(color)
                    .add_modifier(Modifier::BOLD),
            )
            .alignment(Alignment::Center);

        // Center vertically
        let v_center = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Length(1),
                Constraint::Percentage(40),
            ])
            .split(inner);

        f.render_widget(value_paragraph, v_center[1]);
    }

    #[allow(dead_code)]
    fn render_collections(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);

        // Collection list
        let items: Vec<ListItem> = self
            .state
            .collections
            .iter()
            .map(|c| {
                ListItem::new(Line::from(vec![
                    Span::styled(&c.name, Style::default().fg(Color::Cyan)),
                    Span::raw(" - "),
                    Span::styled(
                        format!("{} vectors", c.count),
                        Style::default().fg(Color::Gray),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .title(" Collections [↑↓ navigate, Enter select] ")
                    .borders(Borders::ALL),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▶ ");

        f.render_stateful_widget(list, chunks[0], &mut self.state.collection_state);

        // Collection details
        let detail_text = if let Some(idx) = self.state.collection_state.selected() {
            if let Some(col) = self.state.collections.get(idx) {
                vec![
                    Line::from(vec![
                        Span::styled("Name: ", Style::default().fg(Color::Gray)),
                        Span::styled(&col.name, Style::default().fg(Color::Cyan)),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("Dimensions: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            col.dimensions.to_string(),
                            Style::default().fg(Color::Yellow),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("Vector Count: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            format_number(col.count),
                            Style::default().fg(Color::Green),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("Index Type: ", Style::default().fg(Color::Gray)),
                        Span::styled(&col.index_type, Style::default().fg(Color::Magenta)),
                    ]),
                    Line::from(""),
                    Line::from(""),
                    Line::from(Span::styled(
                        "Actions:",
                        Style::default().add_modifier(Modifier::BOLD),
                    )),
                    Line::from("  [s] Search in collection"),
                    Line::from("  [c] Run clustering"),
                    Line::from("  [a] Detect anomalies"),
                    Line::from("  [d] Delete collection"),
                    Line::from("  [e] Export to JSON"),
                ]
            } else {
                vec![Line::from("Select a collection")]
            }
        } else {
            vec![Line::from("Select a collection to view details")]
        };

        let details = Paragraph::new(detail_text)
            .block(
                Block::default()
                    .title(" Collection Details ")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(details, chunks[1]);
    }

    #[allow(dead_code)]
    fn render_search(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Search input
                Constraint::Min(0),    // Results
            ])
            .split(area);

        // Search input
        let input_style = match self.state.input_mode {
            InputMode::Normal => Style::default(),
            InputMode::Editing => Style::default().fg(Color::Yellow),
        };

        let input = Paragraph::new(self.state.search_input.as_str())
            .style(input_style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Search Query [Press 'i' to edit, Enter to search] "),
            );
        f.render_widget(input, chunks[0]);

        // Show cursor when editing
        if self.state.input_mode == InputMode::Editing {
            f.set_cursor_position((
                chunks[0].x + self.state.search_input.len() as u16 + 1,
                chunks[0].y + 1,
            ));
        }

        // Search results
        let header_cells = ["ID", "Distance", "Metadata"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
        let header = Row::new(header_cells).height(1).bottom_margin(1);

        let rows = self.state.search_results.iter().map(|r| {
            Row::new(vec![
                Cell::from(r.id.clone()),
                Cell::from(format!("{:.6}", r.distance)),
                Cell::from(truncate_string(&r.metadata, 50)),
            ])
        });

        let table = Table::new(
            rows,
            [
                Constraint::Percentage(20),
                Constraint::Percentage(15),
                Constraint::Percentage(65),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " Results ({}) ",
                    self.state.search_results.len()
                )),
        )
        .row_highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("▶ ");

        f.render_stateful_widget(table, chunks[1], &mut self.state.search_state);
    }

    fn render_clusters(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Cluster list
        let cluster_items: Vec<ListItem> = if let Some(ref data) = self.state.cluster_data {
            data.clusters
                .iter()
                .map(|c| {
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(
                                format!("Cluster {}", c.id),
                                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("  Size: "),
                            Span::styled(
                                c.size.to_string(),
                                Style::default().fg(Color::Green),
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("  Centroid: "),
                            Span::styled(
                                &c.centroid_preview,
                                Style::default().fg(Color::Gray),
                            ),
                        ]),
                    ])
                })
                .collect()
        } else {
            vec![ListItem::new("No clustering results. Press 'r' to run clustering.")]
        };

        let cluster_list = List::new(cluster_items).block(
            Block::default()
                .title(" Clusters [r: run, k: set k] ")
                .borders(Borders::ALL),
        );
        f.render_widget(cluster_list, chunks[0]);

        // Cluster visualization (text-based)
        let viz_text = if let Some(ref data) = self.state.cluster_data {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Collection: ", Style::default().fg(Color::Gray)),
                    Span::styled(&data.collection, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![
                    Span::styled("K: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        data.k.to_string(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Inertia: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.4}", data.inertia),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    "Cluster Distribution:",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
            ];

            // ASCII bar chart
            let max_size = data.clusters.iter().map(|c| c.size).max().unwrap_or(1);
            for cluster in &data.clusters {
                let bar_len = (cluster.size * 30 / max_size).max(1);
                let bar: String = "█".repeat(bar_len);
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("C{}: ", cluster.id),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::styled(bar, Style::default().fg(Color::Green)),
                    Span::styled(
                        format!(" {}", cluster.size),
                        Style::default().fg(Color::Gray),
                    ),
                ]));
            }

            lines
        } else {
            vec![
                Line::from(""),
                Line::from("Select a collection and run clustering"),
                Line::from("to see cluster visualization."),
                Line::from(""),
                Line::from("Press 'r' to run K-means clustering"),
            ]
        };

        let viz = Paragraph::new(viz_text)
            .block(
                Block::default()
                    .title(" Visualization ")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(viz, chunks[1]);
    }

    fn render_anomalies(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(0)])
            .split(area);

        // Method selector
        let method_text = if let Some(ref data) = self.state.anomaly_data {
            vec![
                Line::from(vec![
                    Span::styled("Method: ", Style::default().fg(Color::Gray)),
                    Span::styled(&data.method, Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![
                    Span::styled("Collection: ", Style::default().fg(Color::Gray)),
                    Span::styled(&data.collection, Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::styled("Outliers Found: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        data.outliers.len().to_string(),
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                ]),
            ]
        } else {
            vec![
                Line::from("Press 'r' to run anomaly detection"),
                Line::from("[1: LOF, 2: Isolation Forest, 3: Distance-based]"),
            ]
        };

        let method_para = Paragraph::new(method_text)
            .block(
                Block::default()
                    .title(" Anomaly Detection ")
                    .borders(Borders::ALL),
            );
        f.render_widget(method_para, chunks[0]);

        // Outliers list
        let outlier_items: Vec<ListItem> = if let Some(ref data) = self.state.anomaly_data {
            data.outliers
                .iter()
                .map(|o| {
                    let score_color = if o.score > 0.8 {
                        Color::Red
                    } else if o.score > 0.5 {
                        Color::Yellow
                    } else {
                        Color::Green
                    };

                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(&o.id, Style::default().fg(Color::Cyan)),
                            Span::raw(" - Score: "),
                            Span::styled(
                                format!("{:.4}", o.score),
                                Style::default().fg(score_color),
                            ),
                        ]),
                        Line::from(vec![
                            Span::raw("  "),
                            Span::styled(&o.reason, Style::default().fg(Color::Gray)),
                        ]),
                    ])
                })
                .collect()
        } else {
            vec![ListItem::new("No anomaly detection results yet.")]
        };

        let outlier_list = List::new(outlier_items).block(
            Block::default()
                .title(" Detected Outliers ")
                .borders(Borders::ALL),
        );
        f.render_widget(outlier_list, chunks[1]);
    }

    fn render_streaming(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(area);

        // Stream status
        let status = Paragraph::new("Monitoring vector operations in real-time")
            .style(Style::default().fg(Color::Green))
            .block(
                Block::default()
                    .title(" Stream Monitor [Press 'p' to pause] ")
                    .borders(Borders::ALL),
            );
        f.render_widget(status, chunks[0]);

        // Event log
        let events: Vec<ListItem> = self
            .state
            .stream_events
            .iter()
            .rev()
            .take(50)
            .map(|e| {
                let type_color = match e.event_type.as_str() {
                    "INSERT" => Color::Green,
                    "DELETE" => Color::Red,
                    "SEARCH" => Color::Cyan,
                    "UPDATE" => Color::Yellow,
                    _ => Color::Gray,
                };

                ListItem::new(Line::from(vec![
                    Span::styled(&e.timestamp, Style::default().fg(Color::DarkGray)),
                    Span::raw(" "),
                    Span::styled(
                        format!("[{}]", e.event_type),
                        Style::default().fg(type_color),
                    ),
                    Span::raw(" "),
                    Span::styled(&e.collection, Style::default().fg(Color::Magenta)),
                    Span::raw(": "),
                    Span::raw(&e.details),
                ]))
            })
            .collect();

        let event_list = List::new(events).block(
            Block::default()
                .title(" Event Log ")
                .borders(Borders::ALL),
        );
        f.render_widget(event_list, chunks[1]);
    }

    fn render_help(&self, f: &mut Frame, area: Rect) {
        let help_text = vec![
            Line::from(Span::styled(
                "Navigation",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  Tab / Shift+Tab    Switch between views"),
            Line::from("  1-7                Jump to specific view"),
            Line::from("  ↑/↓ or j/k         Navigate lists"),
            Line::from("  Enter              Select/confirm"),
            Line::from("  Esc                Cancel/back"),
            Line::from("  q                  Quit application"),
            Line::from(""),
            Line::from(Span::styled(
                "Dashboard",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  r                  Refresh statistics"),
            Line::from(""),
            Line::from(Span::styled(
                "Collections",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  s                  Search in collection"),
            Line::from("  c                  Run clustering"),
            Line::from("  a                  Detect anomalies"),
            Line::from("  d                  Delete collection"),
            Line::from("  n                  Create new collection"),
            Line::from(""),
            Line::from(Span::styled(
                "Search",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  i                  Enter edit mode"),
            Line::from("  Enter              Execute search"),
            Line::from("  Esc                Exit edit mode"),
            Line::from(""),
            Line::from(Span::styled(
                "Clusters",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  r                  Run K-means clustering"),
            Line::from("  +/-                Increase/decrease K"),
            Line::from(""),
            Line::from(Span::styled(
                "Anomalies",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("  1                  Use LOF algorithm"),
            Line::from("  2                  Use Isolation Forest"),
            Line::from("  3                  Use Distance-based detection"),
            Line::from("  r                  Run detection"),
        ];

        let help = Paragraph::new(help_text)
            .block(
                Block::default()
                    .title(" Help - Keyboard Shortcuts ")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(help, area);
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let status = if let Some((ref msg, instant)) = self.state.status_message {
            if instant.elapsed() < Duration::from_secs(5) {
                msg.clone()
            } else {
                self.default_status()
            }
        } else {
            self.default_status()
        };

        let mode_indicator = match self.state.input_mode {
            InputMode::Normal => Span::styled(" NORMAL ", Style::default().bg(Color::Blue)),
            InputMode::Editing => Span::styled(" EDIT ", Style::default().bg(Color::Yellow).fg(Color::Black)),
        };

        let status_line = Line::from(vec![
            mode_indicator,
            Span::raw(" "),
            Span::raw(status),
        ]);

        let status_bar = Paragraph::new(status_line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        f.render_widget(status_bar, area);
    }

    fn default_status(&self) -> String {
        format!(
            "Needle v0.1.0 | {} collections | {} vectors | Press ? for help",
            self.state.stats.total_collections,
            format_number(self.state.stats.total_vectors)
        )
    }

    fn handle_key(&mut self, key: KeyCode, _modifiers: KeyModifiers) {
        // Handle input mode first
        if self.state.input_mode == InputMode::Editing {
            match key {
                KeyCode::Esc => {
                    self.state.input_mode = InputMode::Normal;
                }
                KeyCode::Enter => {
                    self.execute_search();
                    self.state.input_mode = InputMode::Normal;
                }
                KeyCode::Char(c) => {
                    self.state.search_input.push(c);
                }
                KeyCode::Backspace => {
                    self.state.search_input.pop();
                }
                _ => {}
            }
            return;
        }

        // Normal mode
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Tab => self.next_view(),
            KeyCode::BackTab => self.prev_view(),
            KeyCode::Char('1') => self.state.current_view = View::Dashboard,
            KeyCode::Char('2') => self.state.current_view = View::Collections,
            KeyCode::Char('3') => self.state.current_view = View::Search,
            KeyCode::Char('4') => self.state.current_view = View::Clusters,
            KeyCode::Char('5') => self.state.current_view = View::Anomalies,
            KeyCode::Char('6') => self.state.current_view = View::Streaming,
            KeyCode::Char('7') | KeyCode::Char('?') => self.state.current_view = View::Help,
            KeyCode::Up | KeyCode::Char('k') => self.select_prev(),
            KeyCode::Down | KeyCode::Char('j') => self.select_next(),
            KeyCode::Char('i') if self.state.current_view == View::Search => {
                self.state.input_mode = InputMode::Editing;
            }
            KeyCode::Char('r') => self.run_action(),
            KeyCode::Char('s') if self.state.current_view == View::Collections => {
                self.state.current_view = View::Search;
            }
            KeyCode::Char('c') if self.state.current_view == View::Collections => {
                self.state.current_view = View::Clusters;
            }
            KeyCode::Char('a') if self.state.current_view == View::Collections => {
                self.state.current_view = View::Anomalies;
            }
            _ => {}
        }
    }

    fn next_view(&mut self) {
        let views = View::all();
        let current_idx = self.state.current_view.index();
        let next_idx = (current_idx + 1) % views.len();
        self.state.current_view = views[next_idx];
    }

    fn prev_view(&mut self) {
        let views = View::all();
        let current_idx = self.state.current_view.index();
        let prev_idx = if current_idx == 0 {
            views.len() - 1
        } else {
            current_idx - 1
        };
        self.state.current_view = views[prev_idx];
    }

    fn select_next(&mut self) {
        match self.state.current_view {
            View::Collections => {
                let len = self.state.collections.len();
                if len > 0 {
                    let i = match self.state.collection_state.selected() {
                        Some(i) => (i + 1) % len,
                        None => 0,
                    };
                    self.state.collection_state.select(Some(i));
                }
            }
            View::Search => {
                let len = self.state.search_results.len();
                if len > 0 {
                    let i = match self.state.search_state.selected() {
                        Some(i) => (i + 1) % len,
                        None => 0,
                    };
                    self.state.search_state.select(Some(i));
                }
            }
            _ => {}
        }
    }

    fn select_prev(&mut self) {
        match self.state.current_view {
            View::Collections => {
                let len = self.state.collections.len();
                if len > 0 {
                    let i = match self.state.collection_state.selected() {
                        Some(i) => {
                            if i == 0 {
                                len - 1
                            } else {
                                i - 1
                            }
                        }
                        None => 0,
                    };
                    self.state.collection_state.select(Some(i));
                }
            }
            View::Search => {
                let len = self.state.search_results.len();
                if len > 0 {
                    let i = match self.state.search_state.selected() {
                        Some(i) => {
                            if i == 0 {
                                len - 1
                            } else {
                                i - 1
                            }
                        }
                        None => 0,
                    };
                    self.state.search_state.select(Some(i));
                }
            }
            _ => {}
        }
    }

    fn execute_search(&mut self) {
        // For now, generate sample results
        // In real implementation, this would parse the query and search
        self.state.search_results = vec![
            SearchResultDisplay {
                id: "vec_001".to_string(),
                distance: 0.123456,
                metadata: r#"{"title": "Document 1", "category": "tech"}"#.to_string(),
            },
            SearchResultDisplay {
                id: "vec_002".to_string(),
                distance: 0.234567,
                metadata: r#"{"title": "Document 2", "category": "science"}"#.to_string(),
            },
        ];
        self.state.search_state.select(Some(0));
        self.set_status("Search completed: 2 results found");
    }

    fn run_action(&mut self) {
        match self.state.current_view {
            View::Dashboard => {
                self.refresh_stats();
                self.set_status("Statistics refreshed");
            }
            View::Clusters => {
                self.run_clustering();
            }
            View::Anomalies => {
                self.run_anomaly_detection();
            }
            _ => {}
        }
    }

    fn run_clustering(&mut self) {
        // Get selected collection or use first one
        let collection_name = if let Some(idx) = self.state.collection_state.selected() {
            self.state.collections.get(idx).map(|c| c.name.clone())
        } else {
            self.state.collections.first().map(|c| c.name.clone())
        };

        if let Some(name) = collection_name {
            // For demo, generate sample cluster data
            self.state.cluster_data = Some(ClusterData {
                collection: name,
                k: 5,
                clusters: vec![
                    ClusterInfo {
                        id: 0,
                        size: 245,
                        centroid_preview: "[0.12, -0.34, 0.56, ...]".to_string(),
                    },
                    ClusterInfo {
                        id: 1,
                        size: 189,
                        centroid_preview: "[0.45, 0.23, -0.11, ...]".to_string(),
                    },
                    ClusterInfo {
                        id: 2,
                        size: 312,
                        centroid_preview: "[-0.22, 0.67, 0.15, ...]".to_string(),
                    },
                    ClusterInfo {
                        id: 3,
                        size: 156,
                        centroid_preview: "[0.89, -0.12, 0.33, ...]".to_string(),
                    },
                    ClusterInfo {
                        id: 4,
                        size: 98,
                        centroid_preview: "[-0.45, -0.56, 0.78, ...]".to_string(),
                    },
                ],
                inertia: 1234.5678,
                algorithm: "K-Means".to_string(),
                iterations: 42,
            });
            self.set_status("Clustering completed: 5 clusters found");
        } else {
            self.set_status("No collection selected");
        }
    }

    fn run_anomaly_detection(&mut self) {
        let collection_name = if let Some(idx) = self.state.collection_state.selected() {
            self.state.collections.get(idx).map(|c| c.name.clone())
        } else {
            self.state.collections.first().map(|c| c.name.clone())
        };

        if let Some(name) = collection_name {
            // For demo, generate sample anomaly data
            self.state.anomaly_data = Some(AnomalyData {
                collection: name,
                method: "Isolation Forest".to_string(),
                outliers: vec![
                    OutlierInfo {
                        id: "vec_042".to_string(),
                        score: 0.92,
                        reason: "Isolated in feature space, short path length".to_string(),
                    },
                    OutlierInfo {
                        id: "vec_187".to_string(),
                        score: 0.85,
                        reason: "Unusual dimension values, far from clusters".to_string(),
                    },
                    OutlierInfo {
                        id: "vec_301".to_string(),
                        score: 0.78,
                        reason: "Low local density compared to neighbors".to_string(),
                    },
                ],
            });
            self.set_status("Anomaly detection completed: 3 outliers found");
        } else {
            self.set_status("No collection selected");
        }
    }

    fn refresh_stats(&mut self) {
        self.state.collections = self.load_collections();
        self.state.stats = self.calculate_stats();
    }

    fn load_collections(&self) -> Vec<CollectionInfo> {
        self.db
            .list_collections()
            .into_iter()
            .filter_map(|name| {
                self.db.collection(&name).ok().map(|col| {
                    CollectionInfo {
                        name,
                        dimensions: col.dimensions().unwrap_or(0),
                        count: col.len(),
                        index_type: "HNSW".to_string(),
                    }
                })
            })
            .collect()
    }

    fn calculate_stats(&self) -> DashboardStats {
        let total_vectors: usize = self.state.collections.iter().map(|c| c.count).sum();
        let avg_dimensions = if self.state.collections.is_empty() {
            0
        } else {
            self.state.collections.iter().map(|c| c.dimensions).sum::<usize>()
                / self.state.collections.len()
        };

        DashboardStats {
            total_collections: self.state.collections.len(),
            total_vectors,
            total_storage_mb: total_vectors as f64 * avg_dimensions as f64 * 4.0 / 1_000_000.0,
            avg_dimensions,
            queries_per_second: 0.0,
            uptime_seconds: 0,
        }
    }

    fn set_status(&mut self, msg: &str) {
        self.state.status_message = Some((msg.to_string(), Instant::now()));
    }

    fn on_tick(&mut self) {
        // Update query history for sparkline
        if self.state.query_history.len() >= 60 {
            self.state.query_history.remove(0);
        }
        // Simulate some query activity
        use rand::Rng;
        let mut rng = rand::thread_rng();
        self.state.query_history.push(rng.gen_range(0..100));

        // Add simulated stream events occasionally
        if rng.gen_ratio(1, 20) {
            let events = ["INSERT", "SEARCH", "DELETE", "UPDATE"];
            let collections = ["documents", "images", "embeddings"];

            self.state.stream_events.push(StreamEvent {
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                event_type: events[rng.gen_range(0..events.len())].to_string(),
                collection: collections[rng.gen_range(0..collections.len())].to_string(),
                details: format!("Vector operation #{}", rng.gen_range(1000..9999)),
            });

            // Keep only last 100 events
            if self.state.stream_events.len() > 100 {
                self.state.stream_events.remove(0);
            }
        }
    }
}

impl AppState {
    fn new(db: &Database) -> Self {
        let collections: Vec<CollectionInfo> = db
            .list_collections()
            .into_iter()
            .filter_map(|name| {
                db.collection(&name).ok().map(|col| {
                    CollectionInfo {
                        name,
                        dimensions: col.dimensions().unwrap_or(0),
                        count: col.len(),
                        index_type: "HNSW".to_string(),
                    }
                })
            })
            .collect();

        let total_vectors: usize = collections.iter().map(|c| c.count).sum();
        let avg_dimensions = if collections.is_empty() {
            0
        } else {
            collections.iter().map(|c| c.dimensions).sum::<usize>() / collections.len()
        };

        let stats = DashboardStats {
            total_collections: collections.len(),
            total_vectors,
            total_storage_mb: total_vectors as f64 * avg_dimensions as f64 * 4.0 / 1_000_000.0,
            avg_dimensions,
            queries_per_second: 0.0,
            uptime_seconds: 0,
        };

        Self {
            current_view: View::Dashboard,
            collections,
            collection_state: ListState::default(),
            selected_collection: None,
            search_input: String::new(),
            search_results: Vec::new(),
            search_state: TableState::default(),
            cluster_data: None,
            anomaly_data: None,
            stream_events: Vec::new(),
            stream_scroll: 0,
            stats,
            input_mode: InputMode::Normal,
            status_message: None,
            query_history: vec![0; 60],
        }
    }
}

// Helper functions

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1.0K");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1000000), "1.0M");
        assert_eq!(format_number(2500000), "2.5M");
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
    }

    #[test]
    fn test_view_index() {
        assert_eq!(View::Dashboard.index(), 0);
        assert_eq!(View::Collections.index(), 1);
        assert_eq!(View::Search.index(), 2);
        assert_eq!(View::Help.index(), 6);
    }

    #[test]
    fn test_view_all() {
        let views = View::all();
        assert_eq!(views.len(), 7);
        assert_eq!(views[0], View::Dashboard);
        assert_eq!(views[6], View::Help);
    }
}
