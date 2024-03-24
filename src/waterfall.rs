use wasm_bindgen::prelude::*;

use std::io::Write;

use js_sys::Uint8Array;
use zstd::stream;

#[wasm_bindgen]
pub struct ZstdStreamDecoder {
    decoder: stream::write::Decoder<'static, Vec<u8>>,
}

#[wasm_bindgen]
impl ZstdStreamDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ZstdStreamDecoder {
        let decoder = stream::write::Decoder::new(Vec::new()).unwrap();
        ZstdStreamDecoder { decoder }
    }
    pub fn clear(&mut self) {
        let output = self.decoder.get_mut();
        output.clear();
    }
    pub fn decode(&mut self, input: &[u8]) -> Vec<Uint8Array> {
        self.clear();
        self.decoder.write(input).unwrap();
        let slice = self.decoder.get_ref().as_slice();
        if slice.len() == 0 {
            return Vec::new();
        }
        [Uint8Array::from(slice)].to_vec()
    }
}
