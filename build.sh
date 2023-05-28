export CMAKE=$(pwd)/cmake.sh
export CFLAGS="-I$(pwd)/include -Os -nostdlib -Wl,--no-entry"
wasm-pack build
wasm2js -Os pkg/phantomsdrdsp_bg.wasm -o pkg/phantomsdrdsp_bg_fallback.js

#export RUSTFLAGS="-C target-feature=+simd128"
#export CFLAGS="-I$(pwd)/include -Os -msimd128 -nostdlib -Wl,--no-entry"
#wasm-pack build
