//! Application state and the top-level egui frame (tab bar + active tab).

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use eframe::egui;

use crate::{
    agent::{AgentSession, PlaceholderBackend},
    project::{Project, ProjectKind, ProjectStore},
    settings::Settings,
    shell::ShellSession,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Projects,
    Editor,
    Agent,
    Terminal,
    Settings,
}

impl Tab {
    pub const ALL: [Self; 5] = [
        Self::Projects,
        Self::Editor,
        Self::Agent,
        Self::Terminal,
        Self::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Projects => "Projects",
            Self::Editor => "Editor",
            Self::Agent => "Agent",
            Self::Terminal => "Terminal",
            Self::Settings => "Settings",
        }
    }
}

pub struct ZnApp {
    pub(crate) data_dir: PathBuf,
    pub(crate) settings: Settings,
    pub(crate) store: ProjectStore,
    pub(crate) tab: Tab,

    pub(crate) projects: Vec<Project>,
    pub(crate) current: Option<Project>,
    pub(crate) new_name: String,
    pub(crate) new_kind: ProjectKind,
    /// Last error or notice shown on the Projects/Editor tabs.
    pub(crate) message: Option<String>,

    pub(crate) files: Vec<PathBuf>,
    pub(crate) open_file: Option<PathBuf>,
    pub(crate) buffer: String,
    pub(crate) dirty: bool,
    pub(crate) new_file_name: String,

    pub(crate) agent: AgentSession,
    pub(crate) chat_input: String,

    pub(crate) shell: ShellSession,
    pub(crate) shell_input: String,
}

impl ZnApp {
    pub fn new(cc: &eframe::CreationContext<'_>, data_dir: PathBuf) -> io::Result<Self> {
        let settings = Settings::load(&data_dir);
        apply_settings(&cc.egui_ctx, &settings);
        let store = ProjectStore::new(&data_dir)?;
        let projects = store.list().unwrap_or_default();
        Ok(Self {
            data_dir,
            settings,
            store,
            tab: Tab::Projects,
            projects,
            current: None,
            new_name: String::new(),
            new_kind: ProjectKind::App,
            message: None,
            files: Vec::new(),
            open_file: None,
            buffer: String::new(),
            dirty: false,
            new_file_name: String::new(),
            agent: AgentSession::new(Box::new(PlaceholderBackend)),
            chat_input: String::new(),
            shell: ShellSession::default(),
            shell_input: String::new(),
        })
    }

    pub(crate) fn refresh_projects(&mut self) {
        match self.store.list() {
            Ok(list) => self.projects = list,
            Err(err) => self.message = Some(format!("Could not list projects: {err}")),
        }
    }

    pub(crate) fn create_project(&mut self) {
        match self.store.create(&self.new_name, self.new_kind) {
            Ok(project) => {
                self.new_name.clear();
                self.refresh_projects();
                self.open_project(project);
            }
            Err(err) => self.message = Some(err.to_string()),
        }
    }

    pub(crate) fn open_project(&mut self, project: Project) {
        self.message = None;
        self.open_file = None;
        self.buffer.clear();
        self.dirty = false;
        self.files = self.store.files(&project).unwrap_or_else(|err| {
            self.message = Some(format!("Could not read files: {err}"));
            Vec::new()
        });
        self.current = Some(project);
    }

    pub(crate) fn load_file(&mut self, rel: PathBuf) {
        let Some(project) = &self.current else { return };
        match self.store.read(project, &rel) {
            Ok(text) => {
                self.buffer = text;
                self.open_file = Some(rel);
                self.dirty = false;
                self.message = None;
            }
            Err(err) => self.message = Some(format!("Could not open {}: {err}", rel.display())),
        }
    }

    pub(crate) fn save_file(&mut self) {
        let (Some(project), Some(rel)) = (&self.current, &self.open_file) else {
            return;
        };
        match self.store.write(project, rel, &self.buffer) {
            Ok(()) => {
                self.dirty = false;
                self.message = Some(format!("Saved {}", rel.display()));
            }
            Err(err) => self.message = Some(format!("Could not save: {err}")),
        }
    }

    pub(crate) fn create_file(&mut self) {
        let Some(project) = self.current.clone() else {
            return;
        };
        let rel = PathBuf::from(self.new_file_name.trim());
        if self.store.write(&project, &rel, "").is_err() {
            self.message = Some("Invalid file name".to_owned());
            return;
        }
        self.new_file_name.clear();
        self.files = self.store.files(&project).unwrap_or_default();
        self.load_file(rel);
    }

    /// Directory commands run in: the open project, else the projects root.
    pub(crate) fn working_dir(&self) -> &Path {
        self.current
            .as_ref()
            .map_or(self.store.root(), |p| p.path.as_path())
    }

    pub(crate) fn save_settings(&mut self, ctx: &egui::Context) {
        self.settings.font_scale = self
            .settings
            .font_scale
            .clamp(Settings::MIN_SCALE, Settings::MAX_SCALE);
        apply_settings(ctx, &self.settings);
        if let Err(err) = self.settings.save(&self.data_dir) {
            self.message = Some(format!("Could not save settings: {err}"));
        }
    }
}

fn apply_settings(ctx: &egui::Context, settings: &Settings) {
    ctx.set_theme(if settings.dark_mode {
        egui::ThemePreference::Dark
    } else {
        egui::ThemePreference::Light
    });
    ctx.set_zoom_factor(settings.font_scale);
}

/// A repaint callback that worker threads can call.
pub(crate) fn repaint_handle(ctx: &egui::Context) -> Arc<dyn Fn() + Send + Sync> {
    let ctx = ctx.clone();
    Arc::new(move || ctx.request_repaint())
}

impl eframe::App for ZnApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.agent.poll();
        self.shell.poll();
        if self.agent.is_busy() || self.shell.is_running() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }

        egui::Panel::bottom("tabs").show(ui, |ui| {
            ui.style_mut().spacing.button_padding = egui::vec2(6.0, 10.0);
            ui.columns(Tab::ALL.len(), |columns| {
                for (column, tab) in columns.iter_mut().zip(Tab::ALL) {
                    column.selectable_value(&mut self.tab, tab, tab.label());
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Projects => self.ui_projects(ui),
            Tab::Editor => self.ui_editor(ui),
            Tab::Agent => self.ui_agent(ui),
            Tab::Terminal => self.ui_terminal(ui),
            Tab::Settings => self.ui_settings(ui),
        });
    }
}
