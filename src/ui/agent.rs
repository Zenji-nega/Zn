use eframe::egui;

use crate::{
    agent::Role,
    app::{ZnApp, repaint_handle},
};

impl ZnApp {
    pub(crate) fn ui_agent(&mut self, ui: &mut egui::Ui) {
        ui.heading("Agent");
        let context = self
            .current
            .as_ref()
            .map_or("no project selected", |p| p.name.as_str());
        ui.weak(format!(
            "Backend: {} · Project: {context}",
            self.agent.backend_name()
        ));

        let mut submit = false;
        egui::Panel::bottom("agent_input").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let send = ui.add_enabled(!self.agent.is_busy(), egui::Button::new("Send"));
                let input = ui.add(
                    egui::TextEdit::singleline(&mut self.chat_input)
                        .hint_text("Describe what you want to build")
                        .desired_width(f32::INFINITY),
                );
                let enter = input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if send.clicked() || enter {
                    submit = true;
                    input.request_focus();
                }
            });
        });
        if submit {
            let text = std::mem::take(&mut self.chat_input);
            let repaint = repaint_handle(ui.ctx());
            self.agent
                .send(&text, self.current.clone(), move || repaint());
        }

        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for message in &self.agent.history {
                    let who = match message.role {
                        Role::User => "You",
                        Role::Assistant => "Agent",
                        Role::System => "System",
                    };
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.strong(who);
                        ui.label(&message.text);
                    });
                }
                if self.agent.is_busy() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.weak("Thinking…");
                    });
                }
            });
    }
}
