use std::ptr;

pub struct ExecutableBuffer {
    pub ptr: *mut u8,
    pub size: usize,
}

impl ExecutableBuffer {
    pub fn new(code: &[u8]) -> Result<Self, String> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let alloc_size = ((code.len() + page_size - 1) / page_size) * page_size;

        unsafe {
            // ১. Read + Write পারমিশনে মেমোরি পেজ অ্যালোকেট
            let mem = libc::mmap(
                ptr::null_mut(),
                alloc_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            );

            if mem == libc::MAP_FAILED {
                return Err("JIT-MEM: mmap failed".into());
            }

            // ২. ARMv7 মেশিন কোড মেমোরিতে রাইট করা
            ptr::copy_nonoverlapping(code.as_ptr(), mem as *mut u8, code.len());

            // ৩. মেমোরি পেজ এক্সিকিউটেবল করা (PROT_READ | PROT_EXEC)
            if libc::mprotect(mem, alloc_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(mem, alloc_size);
                return Err("JIT-MEM: mprotect PROT_EXEC failed".into());
            }

            // ৪. Linux Kernel ARM Cacheflush Syscall (Syscall 0xf0002)
            // কোনো বাহ্যিক __clear_cache দরকার নেই, সরাসরি কার্নেল ট্র্যাপ
            let start = mem as usize;
            let end = start + alloc_size;
            libc::syscall(0xf0002, start, end, 0);

            Ok(ExecutableBuffer {
                ptr: mem as *mut u8,
                size: alloc_size,
            })
        }
    }

    pub unsafe fn as_fn<F>(&self) -> F {
        std::mem::transmute_copy(&self.ptr)
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}

#[allow(dead_code)]
pub struct NativeJit;

#[allow(dead_code)]
impl NativeJit {
    /// ARMv7 (32-bit) Native Machine Code Emitter
    /// ABI: extern "C" fn(regs: *mut i64, limit: i32) -> i32
    pub fn compile_hot_loop_add() -> Result<ExecutableBuffer, String> {
        // Native ARMv7 (ARM Mode 32-bit little-endian) Machine Code:
        // Parameters:
        //   r0 = pointer to SVM registers (*mut i64)
        //   r1 = loop limit (e.g. 10)
        //
        // 00: e3a02000    mov r2, #0          ; accum = 0
        // 04: e3a03001    mov r3, #1          ; iter = 1
        // [loop:]
        // 08: e1530001    cmp r3, r1          ; cmp iter, limit
        // 0c: ca000002    bgt exit (+8 bytes) ; exit if iter > limit
        // 10: e0822003    add r2, r2, r3      ; accum += iter
        // 14: e2833001    add r3, r3, #1      ; iter += 1
        // 18: eafffffa    b loop (-24 bytes)
        // [exit:]
        // 1c: e5802020    str r2, [r0, #32]   ; regs[4] lower 32-bit = accum (4 * 8 = 32)
        // 20: e5803018    str r3, [r0, #24]   ; regs[3] lower 32-bit = iter  (3 * 8 = 24)
        // 24: e1a00002    mov r0, r2          ; return value in r0
        // 28: e12fff1e    bx lr               ; return from function to SVM
        let armv7_machine_code: [u8; 44] = [
            0x00, 0x20, 0xa0, 0xe3, // mov r2, #0
            0x01, 0x30, 0xa0, 0xe3, // mov r3, #1
            0x01, 0x00, 0x53, 0xe1, // cmp r3, r1
            0x02, 0x00, 0x00, 0xca, // bgt exit
            0x03, 0x20, 0x82, 0xe0, // add r2, r2, r3
            0x01, 0x30, 0x83, 0xe2, // add r3, r3, #1
            0xfa, 0xff, 0xff, 0xea, // b loop
            0x20, 0x20, 0x80, 0xe5, // str r2, [r0, #32]
            0x18, 0x30, 0x80, 0xe5, // str r3, [r0, #18]
            0x02, 0x00, 0xa0, 0xe1, // mov r0, r2
            0x1e, 0xff, 0x2f, 0xe1, // bx lr
        ];

        ExecutableBuffer::new(&armv7_machine_code)
    }
}
