#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::AhmetKayaifyApp;
#[cfg(target_arch = "wasm32")]
pub use app::worker_entry;
