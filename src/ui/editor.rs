use std::path::PathBuf;

use eframe::egui;

use crate::app::ZnApp;

impl ZnApp {
    pub(crate) fn ui_editor(&mut self, ui: &mut egui::Ui) {
        let Some(project) = self.current.clone() else {
            ui.heading("Editor");
            ui.weak("Open a project from the Projects tab first.");
            return;
        };
        ui.horizontal(|ui| {
            ui.heading(&project.name);
            if ui.button("Preview").clicked() {
                self.open_preview();
            }
        });
        if let Some(note) = &self.preview_note {
            ui.colored_label(ui.visuals().warn_fg_color, note);
        }

        let mut to_open: Option<PathBuf> = None;
        let selected_text = self
            .open_file
            .as_ref()
            .map_or("Choose a file".to_owned(), |p| p.display().to_string());
        egui::ComboBox::from_id_salt("open_file")
            .selected_text(selected_text)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for file in &self.files {
                    let selected = self.open_file.as_ref() == Some(file);
                    if ui
                        .selectable_label(selected, file.display().to_string())
                        .clicked()
                    {
                        to_open = Some(file.clone());
                    }
                }
            });
        if let Some(file) = to_open {
            self.load_file(file);
        }

        ui.horizontal(|ui| {
            let can_save = self.open_file.is_some() && self.dirty;
            if ui
                .add_enabled(can_save, egui::Button::new("Save"))
                .clicked()
            {
                self.save_file();
            }
            ui.add(
                egui::TextEdit::singleline(&mut self.new_file_name)
                    .hint_text("new/file.txt")
                    .desired_width(ui.available_width() - 70.0),
            );
            if ui.button("Add").clicked() {
                self.create_file();
            }
        });
        if let Some(message) = &self.message {
            ui.colored_label(ui.visuals().warn_fg_color, message);
        }
        ui.separator();

        if self.open_file.is_none() {
            ui.weak("Pick a file above, or add a new one.");
            return;
        }
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            let response = ui.add(
                egui::TextEdit::multiline(&mut self.buffer)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(24),
            );
            if response.changed() {
                self.dirty = true;
            }
        });
    }
}
