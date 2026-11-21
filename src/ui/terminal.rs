use eframe::egui;

use crate::app::{ZnApp, repaint_handle};

impl ZnApp {
    pub(crate) fn ui_terminal(&mut self, ui: &mut egui::Ui) {
        ui.heading("Terminal");
        ui.weak(format!("Directory: {}", self.working_dir().display()));

        let mut submit = false;
        egui::Panel::bottom("terminal_input").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    self.shell.clear();
                }
                if ui.button("Paste").clicked()
                    && let Some(text) = super::pasted_text()
                {
                    self.shell_input.push_str(&text);
                }
                let run = ui.add_enabled(!self.shell.is_running(), egui::Button::new("Run"));
                let input = ui.add(
                    egui::TextEdit::singleline(&mut self.shell_input)
                        .font(egui::TextStyle::Monospace)
                        .hint_text("command")
                        .desired_width(f32::INFINITY),
                );
                let enter = input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if run.clicked() || enter {
                    submit = true;
                    input.request_focus();
                }
            });
        });
        if submit {
            let command = std::mem::take(&mut self.shell_input);
            let cwd = self.working_dir().to_path_buf();
            let repaint = repaint_handle(ui.ctx());
            self.shell.run(&command, &cwd, move || repaint());
        }

        egui::ScrollArea::both()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.shell.output {
                    ui.monospace(line);
                }
            });
    }
}
