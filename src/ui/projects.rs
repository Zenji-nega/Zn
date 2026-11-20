use eframe::egui;

use crate::{app::ZnApp, project::ProjectKind};

impl ZnApp {
    pub(crate) fn ui_projects(&mut self, ui: &mut egui::Ui) {
        ui.heading("Projects");
        ui.add_space(4.0);

        ui.add(
            egui::TextEdit::singleline(&mut self.new_name)
                .hint_text("New project name")
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("new_project_kind")
                .selected_text(self.new_kind.label())
                .show_ui(ui, |ui| {
                    for kind in ProjectKind::ALL {
                        ui.selectable_value(&mut self.new_kind, kind, kind.label());
                    }
                });
            if ui.button("Create").clicked() {
                self.create_project();
            }
            if ui.button("Refresh").clicked() {
                self.refresh_projects();
            }
        });

        if let Some(message) = &self.message {
            ui.colored_label(ui.visuals().warn_fg_color, message);
        }
        ui.separator();

        if self.projects.is_empty() {
            ui.weak("No projects yet. Create one above.");
        }
        let mut to_open = None;
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                for project in &self.projects {
                    let selected = self
                        .current
                        .as_ref()
                        .is_some_and(|c| c.path == project.path);
                    let label = format!("{}  ·  {}", project.name, project.kind.label());
                    if ui.selectable_label(selected, label).clicked() {
                        to_open = Some(project.clone());
                    }
                }
            });
        if let Some(project) = to_open {
            self.open_project(project);
        }
    }
}
