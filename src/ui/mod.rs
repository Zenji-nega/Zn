//! One file per tab; each adds an `impl ZnApp` method that draws it.

mod agent;
mod editor;
mod projects;
mod settings;
mod terminal;

use eframe::egui;

/// Puts text on the clipboard. Android's windowing layer has no clipboard, so the system
/// one is set directly; `copy_text` covers desktop.
pub(crate) fn copy_text(ctx: &egui::Context, text: &str) {
    ctx.copy_text(text.to_owned());
    let _ = crate::platform::set_clipboard_text(text);
}

pub(crate) fn pasted_text() -> Option<String> {
    crate::platform::clipboard_text().filter(|t| !t.is_empty())
}

/// A one-line text box with a Paste button (long-press paste is not available in egui on
/// Android). Returns `(box response, whether text was pasted)`.
pub(crate) fn paste_edit(
    ui: &mut egui::Ui,
    value: &mut String,
    secret: bool,
) -> (egui::Response, bool) {
    ui.horizontal(|ui| {
        let pasted = ui.small_button("Paste").clicked()
            && pasted_text().is_some_and(|t| {
                *value = t.trim().to_owned();
                true
            });
        let response = ui.add(
            egui::TextEdit::singleline(value)
                .password(secret)
                .desired_width(f32::INFINITY),
        );
        (response, pasted)
    })
    .inner
}
