use wasm_bindgen::prelude::*;

use std::os::raw::c_void;

use crate::audio::AudioDecoder;

extern "C" {
    pub fn fx_flac_alloc_default() -> *mut c_void;
    pub fn fx_flac_process(
        flac: *mut c_void,
        inbuf: *const u8,
        inlen: *mut u32,
        outbuf: *mut i32,
        outlen: *mut u32,
    ) -> i32;
    pub fn free(ptr: *mut c_void);
}

#[wasm_bindgen]
pub struct FoxenFlacDecoder {
    flac: *mut c_void,
    outbuf: [i32; 16384],
}

#[wasm_bindgen]
impl FoxenFlacDecoder {
    pub fn new() -> FoxenFlacDecoder {
        let flac = unsafe { fx_flac_alloc_default() };
        FoxenFlacDecoder {
            flac,
            outbuf: [0; 16384],
        }
    }
}

impl Drop for FoxenFlacDecoder {
    fn drop(&mut self) {
        unsafe { free(self.flac) };
    }
}

impl AudioDecoder for FoxenFlacDecoder {
    fn decode(&mut self, input: &[u8]) -> Vec<i16> {
        let mut inlen: u32 = input.len() as u32;
        let mut outlen: u32 = 16384;
        let ret = unsafe {
            fx_flac_process(
                self.flac,
                input.as_ptr(),
                &mut inlen,
                self.outbuf.as_mut_ptr(),
                &mut outlen,
            )
        };
        if ret < 0 {
            return Vec::new();
        }
        self.outbuf[..outlen as usize]
            .to_vec()
            .into_iter()
            .map(|x| (x >> 16) as i16)
            .collect()
    }
}
