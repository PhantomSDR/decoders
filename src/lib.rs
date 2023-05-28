mod audio;
mod foxenflac;
mod noiseblankerwild;
mod noisereduction;
mod spectralnoisereduction;
mod stdlib;
mod symphonia;
mod utils;
mod waterfall;

use wasm_bindgen::prelude::*;

use futuredsp::firdes;

use js_sys::Float32Array;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, phantomsdrdsp!");
}
#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
pub fn firdes_kaiser_lowpass(cutoff: f64, transition_bw: f64, max_ripple: f64) -> Float32Array {
    let fir = firdes::kaiser::lowpass::<f32>(cutoff, transition_bw, max_ripple);
    return Float32Array::from(fir.as_slice());
}

/*
#[no_mangle]
pub extern "C" fn console_log_1(s: &str) {
    web_sys::console::log_1(&s.into());
} 

#[no_mangle]
pub extern "C" fn console_log_i8(s: *const i8) {
    let s_str = unsafe { std::ffi::CStr::from_ptr(s).to_str().unwrap() };
    web_sys::console::log_1(&s_str.into());
} 
*/