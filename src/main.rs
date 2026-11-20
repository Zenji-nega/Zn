//! Desktop entry point. Android uses `android_main` in `lib.rs` instead.

fn main() -> eframe::Result {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([420.0, 800.0]),
        ..Default::default()
    };
    zn::run(zn::default_data_dir(), options)
}
