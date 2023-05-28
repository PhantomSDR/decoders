use symphonia::default::codecs::FlacDecoder;
use symphonia::default::formats::FlacReader;
use wasm_bindgen::prelude::*;

use js_sys::Float32Array;

use samplerate::{ConverterType, Samplerate};

use crate::foxenflac::FoxenFlacDecoder;
use crate::noiseblankerwild::NoiseBlankerWild;
use crate::noisereduction::{NoiseReduction, NoiseReductionType};
use crate::spectralnoisereduction::SpectralNoiseReduction;
use crate::symphonia::SymphoniaDecoder;

#[wasm_bindgen]
pub enum AudioCodec {
    Flac,
    Opus,
}

pub trait AudioDecoder {
    fn decode(&mut self, input: &[u8]) -> Vec<i16>;
}

#[wasm_bindgen]
pub struct Audio {
    decoder: Box<dyn AudioDecoder>,
    resampler: Samplerate,
    decoded_callback: Option<js_sys::Function>,

    noise_reduction: NoiseReduction,
    noise_reduction_enabled: bool,
    spectral_noise_reduction: SpectralNoiseReduction,
    noise_blanker: NoiseBlankerWild,
    noise_blanker_enabled: bool,
    autonotch: NoiseReduction,
    autonotch_enabled: bool,
}

#[wasm_bindgen]
impl Audio {
    #[wasm_bindgen(constructor)]
    pub fn new(codec: AudioCodec, _codec_rate: u32, input_rate: f64, output_rate: u32) -> Audio {
        // Get more resolution
        // Find largest power of 2 below 2000000000/input_rate
        let scale = (2000000000. / input_rate).log2().floor() as i32;
        let input_rate_scaled = (input_rate * (2_f64.powi(scale))) as u32;
        let output_rate_scaled = output_rate * (2_u32.pow(scale as u32));
        let resampler = Samplerate::new(
            ConverterType::SincBestQuality,
            input_rate_scaled,
            output_rate_scaled,
            1,
        )
        .expect("resampler");
        let decoder: Box<dyn AudioDecoder> = match codec {
            AudioCodec::Flac => Box::new(SymphoniaDecoder::<FlacReader, FlacDecoder>::new()),
            //AudioCodec::Flac => Box::new(FoxenFlacDecoder::new()),
            AudioCodec::Opus => Box::new(SymphoniaDecoder::<FlacReader, FlacDecoder>::new()),
        };
        let noise_reduction = NoiseReduction::new(
            NoiseReductionType::NoiseReduction,
            64,
            32,
            1.024e-4,
            1.28e-1,
        );
        let spectral_noise_reduction = SpectralNoiseReduction::new(output_rate, 0.0, 0.95, 30.0);
        let noise_blanker = NoiseBlankerWild::new(0.95, 10, 7);
        let autonotch = NoiseReduction::new(NoiseReductionType::Notch, 64, 32, 1.024e-4, 1.28e-1);
        Audio {
            decoder,
            resampler,
            decoded_callback: Option::None,

            noise_reduction,
            noise_reduction_enabled: false,
            spectral_noise_reduction,
            noise_blanker,
            noise_blanker_enabled: false,
            autonotch,
            autonotch_enabled: false,
        }
    }

    pub fn decode(&mut self, input: &[u8]) -> Float32Array {
        let float_decoded = self
            .decoder
            .decode(input)
            .into_iter()
            .map(|x| f32::from(x) / 32768.0)
            .collect::<Vec<f32>>();
        if float_decoded.len() == 0 {
            return Float32Array::new_with_length(0);
        }
        if self.decoded_callback.is_some() {
            let callback = self.decoded_callback.as_mut().unwrap();
            let _ = callback.call1(
                &JsValue::NULL,
                &Float32Array::from(float_decoded.as_slice()),
            );
        }
        let resampled_res = self.resampler.process(&float_decoded);
        if resampled_res.is_err() {
            return Float32Array::new_with_length(0);
        }

        let mut resampled = resampled_res.unwrap();
        if self.noise_reduction_enabled {
            self.spectral_noise_reduction.process(&mut resampled);
        }
        /*if self.noise_reduction_enabled {
            self.noise_reduction.process(&mut resampled);
        }*/
        if self.noise_blanker_enabled {
            self.noise_blanker.process(&mut resampled);
        }
        if self.autonotch_enabled {
            self.autonotch.process(&mut resampled);
        }
        Float32Array::from(resampled.as_slice())
    }

    pub fn set_nr(&mut self, nr: bool) {
        self.noise_reduction_enabled = nr;
    }

    pub fn set_nb(&mut self, nb: bool) {
        self.noise_blanker_enabled = nb;
    }

    pub fn set_an(&mut self, an: bool) {
        self.autonotch_enabled = an;
    }

    pub fn set_decoded_callback(&mut self, f: Option<js_sys::Function>) {
        self.decoded_callback = f;
    }
}
