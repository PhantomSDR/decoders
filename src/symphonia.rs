use std::io::Write;

use ringbuf::{HeapProducer, HeapRb};

use symphonia::core::audio::Signal;
use symphonia::core::codecs::{self, Decoder, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSourceStream, ReadOnlySource};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::audio::AudioDecoder;

pub struct SymphoniaDecoder<T: FormatReader, U: Decoder> {
    undecoded: Vec<u8>,
    stream_prod: HeapProducer<u8>,
    //format: Option<Box<dyn FormatReader>>,
    //decoder: Option<Box<dyn Decoder>>,
    format: Option<T>,
    decoder: Option<U>,
    track_id: u32,
}

impl<T: FormatReader, U: Decoder> SymphoniaDecoder<T, U> {
    pub fn new() -> Self {
        let (stream_prod, _) = HeapRb::<u8>::new(2).split();
        Self {
            undecoded: Vec::new(),
            stream_prod,
            decoder: Option::None,
            format: Option::None,
            track_id: 0,
        }
    }

    pub fn init_codec(&mut self, input: &[u8]) -> bool {
        if self.decoder.is_some() {
            return true;
        }
        self.undecoded.extend_from_slice(input);
        // reset stream
        let (stream_prod, stream_cons) = HeapRb::<u8>::new(1024 * 64).split();
        let source = ReadOnlySource::new(stream_cons);
        let stream = MediaSourceStream::new(Box::new(source), Default::default());
        self.stream_prod = stream_prod;
        self.stream_prod
            .write(self.undecoded.as_slice())
            .expect("write failed");

        let fmt_opts: FormatOptions = Default::default();
        let format = T::try_new(stream, &fmt_opts);
        if format.is_err() {
            return false;
        }
        self.format = format.ok();
        let track = self
            .format
            .as_ref()
            .unwrap()
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != codecs::CODEC_TYPE_NULL)
            .unwrap()
            .clone();

        let dec_opts: DecoderOptions = Default::default();
        let decoder = U::try_new(&track.codec_params, &dec_opts);
        if decoder.is_err() {
            return false;
        }
        self.decoder = decoder.ok();
        self.track_id = track.id;
        self.undecoded.clear();
        true
        /*
        let mut hint = Hint::new();
        let extension = match self.codec {
            codecs::CODEC_TYPE_FLAC => "flac",
            codecs::CODEC_TYPE_OPUS => "opus",
            codecs::CODEC_TYPE_VORBIS => "ogg",
            _ => panic!("unsupported codec"),
        };
        hint.with_extension(extension);
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        let probed = symphonia::default::get_probe().format(&hint, stream, &fmt_opts, &meta_opts);

        if probed.is_err() {
            return false;
        }
        // Get the instantiated format reader.
        let format = probed.unwrap().format;
        // Find the first audio track with a known (decodeable) codec.
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != codecs::CODEC_TYPE_NULL)
            .unwrap()
            .clone();

        // Use the default options for the decoder.

        // Create a decoder for the track.
        self.decoder = Some(
            symphonia::default::get_codecs()
                .make(&track.codec_params, &dec_opts)
                .expect("unsupported codec"),
        );
        self.format = Some(format);

        // Store the track identifier, it will be used to filter packets.
        self.track_id = track.id;
        self.undecoded.clear();
        true */
    }
}

impl<T: FormatReader, U: Decoder> AudioDecoder for SymphoniaDecoder<T, U> {
    fn decode(&mut self, input: &[u8]) -> Vec<i16> {
        if !self.init_codec(input) {
            return Vec::new();
        }
        self.stream_prod.write(input).expect("write failed");
        if self.stream_prod.len() < input.len() * 4 {
            return Vec::new();
        }
        let format = self.format.as_mut().unwrap();
        let decoder = self.decoder.as_mut().unwrap();
        let mut ret = Vec::new();
        while self.stream_prod.len() > input.len() * 2 {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(Error::ResetRequired) => {
                    break;
                }
                Err(_) => {
                    // No more bytes
                    //console::log_2(&"Error: {:?}".into(), &err.to_string().into());
                    break;
                }
            };
            //console::log_2(&"packet".into(), &packet.data.len().into());
            // Consume any new metadata that has been read since the last packet.
            while !format.metadata().is_latest() {
                // Pop the old head of the metadata queue.
                format.metadata().pop();

                // Consume the new metadata at the head of the metadata queue.
            }

            // If the packet does not belong to the selected track, skip over it.
            if packet.track_id() != self.track_id {
                return Vec::new();
            }

            // Decode the packet into audio samples.
            match decoder.decode(&packet) {
                Ok(_decoded) => {
                    // Consume the decoded audio samples (see below).
                    let mut _decoded_i16 = _decoded.make_equivalent::<i16>();
                    _decoded.convert(&mut _decoded_i16);
                    ret.extend_from_slice(_decoded_i16.chan(0));
                }
                Err(Error::IoError(_)) => {
                    // The packet failed to decode due to an IO error, skip the packet.
                    panic!("IO Error");
                    //return Vec::new();
                }
                Err(Error::DecodeError(_)) => {
                    // The packet failed to decode due to invalid data, skip the packet.
                    panic!("Decode Error");
                    //return Vec::new();
                }
                Err(err) => {
                    // An unrecoverable error occured, halt decoding.
                    panic!("{}", err);
                }
            }
        }
        ret
    }
}
