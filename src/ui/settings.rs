use eframe::egui;

use crate::{
    app::ZnApp,
    settings::{ProviderKind, Settings},
};

impl ZnApp {
    pub(crate) fn ui_settings(&mut self, ui: &mut egui::Ui) {
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
    ui.add(
        egui::TextEdit::singleline(value)
            .password(secret)
            .desired_width(f32::INFINITY),
    )
    .lost_focus()
}
