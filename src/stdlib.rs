use std::alloc::{alloc, dealloc, Layout};
use std::os::raw::{c_double, c_long, c_void};

#[no_mangle]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 1);
        alloc(layout).cast()
    }
}

#[no_mangle]
pub extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size * nmemb, 1);
        let ptr = alloc(layout).cast();
        std::ptr::write_bytes::<u8>(ptr as *mut u8, 0, size * nmemb);
        ptr
    }
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    // layout is not actually used
    let layout = Layout::from_size_align_unchecked(1, 1);
    dealloc(ptr.cast(), layout);
}

#[no_mangle]
pub unsafe extern "C" fn lrint(x: c_double) -> c_long {
    x.round() as c_long
}
