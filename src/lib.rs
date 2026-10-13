#![allow(
    clippy::needless_range_loop,
    clippy::upper_case_acronyms,
    clippy::len_without_is_empty
)]
#[cfg(target_arch = "arm")]
pub mod cxx_bridge;
pub mod gguf_loader;
#[cfg(target_arch = "arm")]
pub mod neural_ffi;
pub mod tokenizer;
