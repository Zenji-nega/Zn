//! Application state and the top-level egui frame (tab bar + active tab).

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use eframe::egui;

use crate::{
    agent::{AgentSession, PollOutcome},
    filepicker::FilePicker,
    llm::{Anthropic, Http, OpenAiCompat, Provider, UreqHttp},
    local::{
        LocalHost, LocalProvider,
        engine::EngineConfig,
        models::{self, Download, ExternalModel, ModelEntry},
    },
    project::{Project, ProjectKind, ProjectStore},
    secrets::Secrets,
    settings::{ProviderKind, Settings},
    shell::ShellSession,
    tools::Toolbox,
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

/// The "choose a model file" screen and what the user has picked so far.
pub struct PickerState {
    pub picker: FilePicker,
    pub chosen: Option<PathBuf>,
    pub name: String,
}

/// Form fields for downloading any GGUF model by link.
#[derive(Default)]
pub struct CustomModel {
    pub name: String,
    pub gguf_url: String,
    pub tokenizer_url: String,
}

pub struct ZnApp {
    pub(crate) data_dir: PathBuf,
    pub(crate) settings: Settings,
    pub(crate) secrets: Secrets,
    pub(crate) local_host: Arc<LocalHost>,
    pub(crate) local_models: Vec<ModelEntry>,
    pub(crate) download: Option<Download>,
    /// Result of the last download attempt, shown in Settings.
    pub(crate) download_note: Option<String>,
    pub(crate) custom_model: CustomModel,
    pub(crate) picker: Option<PickerState>,
    /// Show the "allow access to files" explanation.
    pub(crate) storage_prompt: bool,
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
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        data_dir: PathBuf,
        previous_session: Option<String>,
    ) -> io::Result<Self> {
        let settings = Settings::load(&data_dir);
        let secrets = Secrets::load(&data_dir);
        apply_settings(&cc.egui_ctx, &settings);
        let store = ProjectStore::new(&data_dir)?;
        let projects = store.list().unwrap_or_default();
        let mut app = Self {
            data_dir,
            settings,
            secrets,
            local_host: LocalHost::new(),
            local_models: Vec::new(),
            download: None,
            download_note: None,
            custom_model: CustomModel::default(),
            picker: None,
            storage_prompt: false,
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
            agent: AgentSession::new(),
            chat_input: String::new(),
            shell: ShellSession::default(),
            shell_input: String::new(),
        };
        app.refresh_local_models();
        if let Some(report) = previous_session {
            app.agent.note(report);
        }
        Ok(app)
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

    pub(crate) fn save_secrets(&mut self) {
        if let Err(err) = self.secrets.save(&self.data_dir) {
            self.message = Some(format!("Could not save keys: {err}"));
        }
    }

    /// Builds the model client from the current settings, or explains what is missing.
    pub(crate) fn build_provider(&self) -> Result<Box<dyn Provider>, String> {
        let http: Arc<dyn Http> = Arc::new(UreqHttp::new());
        let s = &self.settings;
        match s.provider {
            ProviderKind::Anthropic => {
                if self.secrets.anthropic_key.trim().is_empty() {
                    return Err("Add your Anthropic API key in Settings first.".into());
                }
                if s.anthropic_model.trim().is_empty() {
                    return Err("Set a model name in Settings first.".into());
                }
                Ok(Box::new(Anthropic {
                    http,
                    api_key: self.secrets.anthropic_key.trim().to_owned(),
                    model: s.anthropic_model.trim().to_owned(),
                    base_url: Anthropic::DEFAULT_BASE_URL.to_owned(),
                }))
            }
            ProviderKind::OnDevice => {
                let model = models::all_models(&self.data_dir, &s.external_models)
                    .into_iter()
                    .find(|m| m.id == s.local_model)
                    .ok_or("Download or choose an on-device model in Settings first.")?;
                if !model.gguf.is_file() {
                    return Err(format!(
                        "The model file was not found: {}. It may have been moved, or the app lost access to it.",
                        model.gguf.display()
                    ));
                }
                Ok(Box::new(LocalProvider::new(
                    Arc::clone(&self.local_host),
                    model,
                    EngineConfig::default(),
                )))
            }
            ProviderKind::OpenAiCompatible => {
                if s.openai_model.trim().is_empty() {
                    return Err("Set a model name in Settings first.".into());
                }
                if s.openai_base_url.trim().is_empty() {
                    return Err("Set the server address in Settings first.".into());
                }
                Ok(Box::new(OpenAiCompat {
                    http,
                    api_key: self.secrets.openai_key.trim().to_owned(),
                    model: s.openai_model.trim().to_owned(),
                    base_url: s.openai_base_url.trim().to_owned(),
                }))
            }
        }
    }

    /// Sends the chat input to the agent.
    pub(crate) fn send_to_agent(&mut self, ctx: &egui::Context) {
        let text = std::mem::take(&mut self.chat_input);
        if text.trim().is_empty() {
            return;
        }
        match self.build_provider() {
            Ok(provider) => {
                let toolbox = Toolbox::new(
                    self.store.clone(),
                    self.current.clone(),
                    self.settings.allow_commands,
                );
                let repaint = repaint_handle(ctx);
                self.agent.send(&text, provider, toolbox, move || repaint());
            }
            Err(problem) => {
                self.chat_input = text;
                self.agent.note(problem);
            }
        }
    }

    /// Opens the file chooser, or explains how to allow file access first.
    pub(crate) fn open_model_picker(&mut self) {
        if crate::platform::has_storage_access() {
            self.storage_prompt = false;
            self.picker = Some(PickerState {
                picker: FilePicker::open(&crate::platform::browse_start()),
                chosen: None,
                name: String::new(),
            });
        } else {
            self.storage_prompt = true;
        }
    }

    /// Remembers a model file the user picked and selects it.
    pub(crate) fn add_external_model(&mut self, gguf: PathBuf, name: &str) -> Result<(), String> {
        let name = name.trim().replace(' ', "-");
        if name.is_empty() {
            return Err("Give the model a name.".into());
        }
        let id = format!("{}{name}", ExternalModel::ID_PREFIX);
        if self.local_models.iter().any(|m| m.id == id) {
            return Err("A model with that name already exists.".into());
        }
        self.settings.external_models.push(ExternalModel {
            id: id.clone(),
            gguf,
        });
        self.settings.local_model = id;
        self.local_host.unload();
        let _ = self.settings.save(&self.data_dir);
        self.refresh_local_models();
        Ok(())
    }

    /// Forgets a picked model file (the file itself is not touched).
    pub(crate) fn remove_external_model(&mut self, id: &str) {
        self.settings.external_models.retain(|m| m.id != id);
        if self.settings.local_model == id {
            self.settings.local_model.clear();
            self.local_host.unload();
        }
        let _ = self.settings.save(&self.data_dir);
        self.refresh_local_models();
    }

    pub(crate) fn refresh_local_models(&mut self) {
        self.local_models = models::all_models(&self.data_dir, &self.settings.external_models);
        if self.settings.local_model.is_empty()
            && let Some(first) = self.local_models.first()
        {
            self.settings.local_model = first.id.clone();
        }
    }

    /// Starts downloading a model unless one is already downloading.
    pub(crate) fn begin_download(
        &mut self,
        ctx: &egui::Context,
        id: &str,
        gguf_url: &str,
        tokenizer_url: &str,
    ) {
        if self.download.is_some() {
            return;
        }
        let repaint = repaint_handle(ctx);
        match models::start_download(&self.data_dir, id, gguf_url, tokenizer_url, move || {
            repaint()
        }) {
            Ok(job) => {
                self.download_note = None;
                self.download = Some(job);
            }
            Err(problem) => self.download_note = Some(problem),
        }
    }

    fn poll_download(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.download else { return };
        ctx.request_repaint_after(Duration::from_millis(250));
        let Some(result) = job.poll() else { return };
        let id = job.id.clone();
        self.download = None;
        self.download_note = Some(match result {
            Ok(()) => format!("Downloaded \"{id}\"."),
            Err(err) if err == "cancelled" => {
                "Download cancelled. Start it again to resume.".to_owned()
            }
            Err(err) => format!("Download failed: {err}"),
        });
        self.refresh_local_models();
        if self.local_models.iter().any(|m| m.id == id) {
            self.settings.local_model = id;
            let _ = self.settings.save(&self.data_dir);
        }
    }

    fn apply_agent_outcome(&mut self, outcome: PollOutcome) {
        if let Some(project) = outcome.project {
            self.refresh_projects();
            self.open_project(project);
        }
        if outcome.files_touched {
            let Some(project) = self.current.clone() else {
                return;
            };
            self.files = self.store.files(&project).unwrap_or_default();
            // Show the agent's edits, unless the user has unsaved changes in the editor.
            if let Some(file) = self.open_file.clone()
                && !self.dirty
                && let Ok(text) = self.store.read(&project, &file)
            {
                self.buffer = text;
            }
        }
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
        let outcome = self.agent.poll();
        self.poll_download(ui.ctx());
        self.apply_agent_outcome(outcome);
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
