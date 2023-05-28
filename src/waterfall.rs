use wasm_bindgen::prelude::*;

use std::io::Write;

use js_sys::Uint8Array;
use zstd::stream;

#[wasm_bindgen]
pub struct ZstdWaterfallDecoder {
    decoder: stream::write::Decoder<'static, Vec<u8>>,
}

#[wasm_bindgen]
impl ZstdWaterfallDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ZstdWaterfallDecoder {
        let decoder = stream::write::Decoder::new(Vec::new()).unwrap();
        ZstdWaterfallDecoder { decoder }
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
        [Uint8Array::from(self.decoder.get_ref().as_slice())].to_vec()
    }
}
