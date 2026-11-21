use eframe::egui;

use crate::{
    app::ZnApp,
    local::models,
    settings::{ProviderKind, Settings},
};

impl ZnApp {
    pub(crate) fn ui_settings(&mut self, ui: &mut egui::Ui) {
        // Everything scrolls: on a phone the model manager is taller than the screen.
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                if self.picker.is_some() {
                    self.ui_picker(ui);
                } else {
                    self.ui_settings_body(ui);
                }
            });
    }

    fn ui_settings_body(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.add_space(4.0);

        let mut changed = false;
        let mut changed_ai = false;
        let mut changed_keys = false;
        changed |= ui
            .checkbox(&mut self.settings.dark_mode, "Dark mode")
            .changed();

        ui.label("Interface size");
        let slider = egui::Slider::new(
            &mut self.settings.font_scale,
            Settings::MIN_SCALE..=Settings::MAX_SCALE,
        );
        // Apply on release: changing the zoom while dragging would move the slider away.
        let response = ui.add(slider);
        changed |= response.drag_stopped() || (response.changed() && !response.dragged());

        ui.label("GitHub username");
        changed |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.github_user)
                    .desired_width(f32::INFINITY),
            )
            .lost_focus();

        if changed {
            let ctx = ui.ctx().clone();
            self.save_settings(&ctx);
        }

        ui.separator();
        ui.heading("AI model");
        egui::ComboBox::from_id_salt("provider")
            .selected_text(self.settings.provider.label())
            .show_ui(ui, |ui| {
                for kind in ProviderKind::ALL {
                    changed_ai |= ui
                        .selectable_value(&mut self.settings.provider, kind, kind.label())
                        .changed();
                }
            });
        match self.settings.provider {
            ProviderKind::Anthropic => {
                ui.label("Model");
                changed_ai |= text_field(ui, &mut self.settings.anthropic_model, false);
                ui.label("API key");
                changed_keys |= text_field(ui, &mut self.secrets.anthropic_key, true);
            }
            ProviderKind::OnDevice => {
                changed_ai |= self.ui_on_device(ui);
            }
            ProviderKind::OpenAiCompatible => {
                ui.label("Server address");
                changed_ai |= text_field(ui, &mut self.settings.openai_base_url, false);
                ui.label("Model");
                changed_ai |= text_field(ui, &mut self.settings.openai_model, false);
                ui.label("API key (optional for local servers)");
                changed_keys |= text_field(ui, &mut self.secrets.openai_key, true);
            }
        }
        changed_ai |= ui
            .checkbox(
                &mut self.settings.allow_commands,
                "Let the agent run shell commands",
            )
            .changed();
        if self.settings.allow_commands {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "The agent can run any command with this app's permissions, without asking each time.",
            );
        }
        ui.weak("Keys are stored only on this device, in the app's private folder.");
        if changed_ai {
            let ctx = ui.ctx().clone();
            self.save_settings(&ctx);
        }
        if changed_keys {
            self.save_secrets();
        }
        if let Some(message) = &self.message {
            ui.colored_label(ui.visuals().warn_fg_color, message);
        }

        ui.separator();
        ui.weak(format!("Data folder: {}", self.data_dir.display()));
        ui.weak(format!("Zn {}", env!("CARGO_PKG_VERSION")));
    }
}

/// A full-width text box; returns true when editing finished (focus lost).
fn text_field(ui: &mut egui::Ui, value: &mut String, secret: bool) -> bool {
    let (response, pasted) = super::paste_edit(ui, value, secret);
    response.lost_focus() || pasted
}

impl ZnApp {
    /// The on-device model manager. Returns true when the chosen model changed.
    fn ui_on_device(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.weak(
            "Runs on this phone with no internet after the download. Small models are slow and \
             less reliable at using tools than cloud models.",
        );

        if self.local_models.is_empty() {
            ui.label("No model installed yet. Download one below.");
        }
        let mut delete: Option<(String, bool)> = None;
        for model in &self.local_models {
            ui.horizontal(|ui| {
                let size = format!("{:.0} MB", model.size_bytes as f64 / 1_048_576.0);
                changed |= ui
                    .radio_value(
                        &mut self.settings.local_model,
                        model.id.clone(),
                        format!("{} · {size}", model.id),
                    )
                    .changed();
                let label = if model.external { "Remove" } else { "Delete" };
                if ui.small_button(label).clicked() {
                    delete = Some((model.id.clone(), model.external));
                }
                if model.external && model.size_bytes == 0 {
                    ui.colored_label(ui.visuals().warn_fg_color, "file not found");
                }
            });
        }
        if let Some((id, external)) = delete {
            if external {
                self.remove_external_model(&id);
                changed = true;
            } else {
                if self.settings.local_model == id {
                    self.local_host.unload();
                    self.settings.local_model.clear();
                    changed = true;
                }
                if let Err(err) = models::delete_model(&self.data_dir, &id) {
                    self.download_note = Some(format!("Could not delete: {err}"));
                }
                self.refresh_local_models();
            }
        }
        ui.weak(format!(
            "Available memory: {}",
            crate::diag::memory_summary().trim_start_matches("RAM ")
        ));
        ui.horizontal(|ui| {
            let loaded = self.local_host.is_loaded();
            if ui
                .add_enabled(loaded, egui::Button::new("Free memory"))
                .clicked()
            {
                self.local_host.unload();
            }
            ui.weak(if loaded {
                "Model is loaded in memory."
            } else {
                "Model is not loaded."
            });
        });

        ui.separator();
        ui.label("Use a model file already on this device");
        ui.weak("Pick a .gguf file from your storage (Qwen2 or Qwen3 models). It is used where it is, not copied.");
        if ui.button("Choose a model file…").clicked() {
            self.open_model_picker();
        }
        if self.storage_prompt {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "To read files from your storage, Android needs you to allow \"All files access\" for this app.",
            );
            ui.horizontal_wrapped(|ui| {
                if ui.button("Allow access to files").clicked()
                    && let Err(err) = crate::platform::request_storage_access()
                {
                    self.download_note = Some(format!("Could not open the settings page: {err}"));
                }
                if ui.button("I allowed it, continue").clicked() {
                    self.open_model_picker();
                }
            });
        }

        ui.separator();
        ui.label("Download a model");
        let installed: Vec<String> = self.local_models.iter().map(|m| m.id.clone()).collect();
        let busy = self.download.is_some();
        let mut start: Option<(String, String, String)> = None;
        for preset in &models::PRESETS {
            ui.horizontal_wrapped(|ui| {
                let have = installed.iter().any(|id| id == preset.id);
                let label = format!("{} · ~{} MB", preset.title, preset.approx_mb);
                ui.label(label);
                let text = if have { "Installed" } else { "Download" };
                if ui
                    .add_enabled(!have && !busy, egui::Button::new(text))
                    .clicked()
                {
                    start = Some((
                        preset.id.into(),
                        preset.gguf_url.into(),
                        preset.tokenizer_url.into(),
                    ));
                }
            });
        }
        egui::CollapsingHeader::new("Download by link").show(ui, |ui| {
            ui.weak("Any Qwen2 or Qwen3 GGUF model (for example from Hugging Face), plus its tokenizer.json.");
            ui.label("Name");
            super::paste_edit(ui, &mut self.custom_model.name, false);
            ui.label("Model link (.gguf)");
            super::paste_edit(ui, &mut self.custom_model.gguf_url, false);
            ui.label("Tokenizer link (tokenizer.json)");
            super::paste_edit(ui, &mut self.custom_model.tokenizer_url, false);
            if ui.add_enabled(!busy, egui::Button::new("Download")).clicked() {
                let c = &self.custom_model;
                start = Some((c.name.trim().into(), c.gguf_url.trim().into(), c.tokenizer_url.trim().into()));
            }
        });
        if let Some((id, gguf, tokenizer)) = start {
            let ctx = ui.ctx().clone();
            self.begin_download(&ctx, &id, &gguf, &tokenizer);
        }

        if let Some(job) = &self.download {
            let bytes = job.bytes.load(std::sync::atomic::Ordering::Relaxed);
            let total = job.total.load(std::sync::atomic::Ordering::Relaxed);
            ui.horizontal(|ui| {
                if total > 0 {
                    let fraction = (bytes as f64 / total as f64) as f32;
                    ui.add(egui::ProgressBar::new(fraction).text(format!(
                        "{} {:.0} / {:.0} MB",
                        job.id,
                        bytes as f64 / 1_048_576.0,
                        total as f64 / 1_048_576.0
                    )));
                } else {
                    ui.spinner();
                    ui.label(format!("{} … {:.0} MB", job.id, bytes as f64 / 1_048_576.0));
                }
                if ui.button("Cancel").clicked() {
                    job.cancel();
                }
            });
        }
        if let Some(note) = &self.download_note {
            ui.colored_label(ui.visuals().warn_fg_color, note);
        }
        changed
    }
}

impl ZnApp {
    /// The "choose a model file" screen.
    fn ui_picker(&mut self, ui: &mut egui::Ui) {
        let Some(state) = self.picker.as_mut() else {
            return;
        };
        let mut close = false;
        let mut add: Option<(std::path::PathBuf, String)> = None;
        let mut note: Option<String> = None;

        ui.heading("Choose a model file");
        if let Some(gguf) = state.chosen.clone() {
            // Step 2: confirm and name it.
            ui.label(gguf.display().to_string());
            let entry = models::ExternalModel {
                id: String::new(),
                gguf: gguf.clone(),
            }
            .entry();
            ui.weak(if entry.tokenizer.is_some() {
                "A tokenizer.json next to the file will be used."
            } else {
                "The tokenizer built into the file will be used."
            });
            ui.label("Name");
            ui.add(egui::TextEdit::singleline(&mut state.name).desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                if ui.button("Add").clicked() {
                    add = Some((gguf, state.name.clone()));
                }
                if ui.button("Back").clicked() {
                    state.chosen = None;
                }
            });
        } else {
            // Step 1: browse.
            ui.horizontal_wrapped(|ui| {
                for (label, path) in crate::platform::shortcuts() {
                    if ui.small_button(label).clicked() {
                        state.picker.go(path);
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Up").clicked() {
                    state.picker.up();
                }
                let response = ui.add(
                    egui::TextEdit::singleline(&mut state.picker.path_text)
                        .desired_width(f32::INFINITY),
                );
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    state.picker.go_to_text();
                }
            });
            if let Some(error) = &state.picker.error {
                ui.colored_label(ui.visuals().warn_fg_color, error);
            } else if state.picker.entries.is_empty() {
                ui.weak("No folders or .gguf files here.");
            }
            ui.separator();
            let mut enter: Option<String> = None;
            let mut pick: Option<String> = None;
            for entry in &state.picker.entries {
                if entry.is_dir {
                    if ui
                        .selectable_label(false, format!("> {}", entry.name))
                        .clicked()
                    {
                        enter = Some(entry.name.clone());
                    }
                } else {
                    let label = format!(
                        "{}  ·  {:.0} MB",
                        entry.name,
                        entry.size as f64 / 1_048_576.0
                    );
                    if ui.selectable_label(false, label).clicked() {
                        pick = Some(entry.name.clone());
                    }
                }
            }
            if let Some(name) = enter {
                state.picker.enter(&name);
            }
            if let Some(name) = pick {
                let path = state.picker.file_path(&name);
                state.name = models::ExternalModel::suggest_name(
                    &path,
                    &self
                        .local_models
                        .iter()
                        .map(|m| m.id.clone())
                        .collect::<Vec<_>>(),
                );
                state.chosen = Some(path);
            }
        }
        ui.separator();
        if ui.button("Cancel").clicked() {
            close = true;
        }

        if let Some((gguf, name)) = add {
            match self.add_external_model(gguf, &name) {
                Ok(()) => close = true,
                Err(problem) => note = Some(problem),
            }
        }
        if let Some(problem) = note {
            self.download_note = Some(problem);
        }
        if close {
            self.picker = None;
        }
    }
}
