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

impl crate::app::ZnApp {
    /// Full-screen read-only text where a part of it can be selected by touch and copied.
    pub(crate) fn ui_select(&mut self, ui: &mut egui::Ui) {
        let Some(text) = self.select_text.clone() else {
            return;
        };
        ui.horizontal(|ui| {
            if ui.button("Back").clicked() {
                self.select_text = None;
            }
            if ui.button("Copy all").clicked() {
                copy_text(ui.ctx(), &text);
            }
            let range = self.select_range.clone();
            if ui
                .add_enabled(range.start < range.end, egui::Button::new("Copy selection"))
                .clicked()
            {
                copy_text(ui.ctx(), &char_slice(&text, range));
            }
        });
        ui.weak("Drag over the text to select, then tap Copy selection.");
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                let mut shown: &str = &text;
                let output = egui::TextEdit::multiline(&mut shown)
                    .desired_width(f32::INFINITY)
                    .show(ui);
                if let Some(range) = output.cursor_range {
                    let sorted = range.as_sorted_char_range();
                    if sorted.start != sorted.end {
                        self.select_range = sorted.start.0..sorted.end.0;
                    }
                }
            });
    }
}

/// The characters in `range` (character positions, not bytes), clamped to the text.
pub(crate) fn char_slice(text: &str, range: std::ops::Range<usize>) -> String {
    text.chars()
        .skip(range.start)
        .take(range.end.saturating_sub(range.start))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::char_slice;

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn slices_by_characters_not_bytes() {
        assert_eq!(char_slice("héllo wörld", 1..4), "éll");
        assert_eq!(char_slice("مرحبا", 1..3), "رح");
        assert_eq!(char_slice("abc", 2..99), "c");
        assert_eq!(char_slice("abc", 3..1), "");
    }
}
