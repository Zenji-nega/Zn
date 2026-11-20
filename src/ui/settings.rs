use eframe::egui;

use crate::{app::ZnApp, settings::Settings};

impl ZnApp {
    pub(crate) fn ui_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.add_space(4.0);

        let mut changed = false;
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
        if let Some(message) = &self.message {
            ui.colored_label(ui.visuals().warn_fg_color, message);
        }

        ui.separator();
        ui.weak(format!("Data folder: {}", self.data_dir.display()));
        ui.weak(format!("Zn {}", env!("CARGO_PKG_VERSION")));
    }
}
