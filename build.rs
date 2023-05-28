use cc;
fn main() {
    println!("cargo:rerun-if-changed=lib/flac.c");
    cc::Build::new()
        .file("lib/flac.c")
        .archiver("llvm-ar")
        .compile("flac");
}
