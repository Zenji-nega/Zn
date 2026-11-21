use eframe::egui;

use crate::{agent::ChatEntry, app::ZnApp};

impl ZnApp {
    pub(crate) fn ui_agent(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Agent");
            if ui
                .add_enabled(!self.agent.is_busy(), egui::Button::new("New chat"))
                .clicked()
            {
                self.agent.clear();
            }
            if ui
                .add_enabled(self.current.is_some(), egui::Button::new("Preview"))
                .on_hover_text("Open the project's website in your browser")
                .clicked()
            {
                self.open_preview();
            }
        });
        if let Some(note) = &self.preview_note {
            ui.colored_label(ui.visuals().warn_fg_color, note);
        }
        let project = self.current.as_ref().map_or("none", |p| p.name.as_str());
        let model = match self.settings.provider {
            crate::settings::ProviderKind::Anthropic => &self.settings.anthropic_model,
            crate::settings::ProviderKind::OpenAiCompatible => &self.settings.openai_model,
            crate::settings::ProviderKind::OnDevice => &self.settings.local_model,
        };
        ui.weak(format!(
            "{} · {model} · project: {project}",
            self.settings.provider.label()
        ));

        let mut submit = false;
        egui::Panel::bottom("agent_input").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if self.agent.is_busy() {
                    if ui.button("Stop").clicked() {
                        self.agent.stop();
                    }
                } else if ui.button("Send").clicked() {
                    submit = true;
                }
                if ui.button("Paste").clicked()
                    && let Some(text) = super::pasted_text()
                {
                    self.chat_input.push_str(&text);
                }
                let input = ui.add_enabled(
                    !self.agent.is_busy(),
                    egui::TextEdit::singleline(&mut self.chat_input)
                        .hint_text("Describe what you want to build")
                        .desired_width(f32::INFINITY),
                );
                if input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    submit = true;
                    input.request_focus();
                }
            });
        });
        if submit {
            let ctx = ui.ctx().clone();
            self.send_to_agent(&ctx);
        }

        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.agent.entries.is_empty() {
                    ui.weak("Describe an app, game or website and the agent will build it in your project.");
                }
                for entry in &self.agent.entries {
                    match entry {
                        ChatEntry::User(text) => bubble(ui, "You", text),
                        ChatEntry::Assistant(text) => bubble(ui, "Agent", text),
                        ChatEntry::System(text) => {
                            ui.colored_label(ui.visuals().warn_fg_color, text);
                        }
                        ChatEntry::Tool { summary, ok, detail } => {
                            let mark = if *ok { "[ok]" } else { "[failed]" };
                            egui::CollapsingHeader::new(format!("{mark} {summary}"))
                                .id_salt(summary.as_str().to_owned() + detail)
                                .show(ui, |ui| {
                                    if ui.small_button("Copy").clicked() {
                                        super::copy_text(ui.ctx(), detail);
                                    }
                                    ui.monospace(detail);
                                });
                        }
                    }
                }
                if self.agent.is_busy() {
                    let status = self.local_host.progress.describe();
                    ui.weak(status.unwrap_or_else(|| "Working…".to_owned()));
                }
            });
    }
}

fn bubble(ui: &mut egui::Ui, who: &str, text: &str) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.strong(who);
            if ui.small_button("Copy").clicked() {
                super::copy_text(ui.ctx(), text);
            }
        });
        ui.label(text);
    });
}
