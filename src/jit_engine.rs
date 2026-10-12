use std::ptr;

// POSIX mmap flags for dynamic executable memory allocation
const PROT_READ: libc::c_int = 0x1;
const PROT_WRITE: libc::c_int = 0x2;
const PROT_EXEC: libc::c_int = 0x4;
const MAP_ANONYMOUS: libc::c_int = 0x20;
const MAP_PRIVATE: libc::c_int = 0x02;

pub struct ExecutableBuffer {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableBuffer {
    pub fn new(code: &[u8]) -> Self {
        let page_size = 4096;
        let alloc_size = ((code.len() + page_size - 1) / page_size) * page_size;

        unsafe {
            // 1. Allocate writable memory page
            let addr = libc::mmap(
                ptr::null_mut(),
                alloc_size,
                PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1,
                0,
            );

            assert!(addr != libc::MAP_FAILED, "Failed to allocate executable memory buffer");

            // 2. Write raw ARMv7 assembly opcodes into the buffer
            ptr::copy_nonoverlapping(code.as_ptr(), addr as *mut u8, code.len());

            // 3. Flip page protection to READ + EXECUTE (No Write = W^X Security)
            let prot_res = libc::mprotect(addr, alloc_size, PROT_READ | PROT_EXEC);
            assert_eq!(prot_res, 0, "Failed to set mprotect to RX");

            // 4. Clear ARM CPU Instruction Cache (Crucial on ARM architectures)
            #[cfg(target_arch = "arm")]
            {
                let start = addr as usize;
                let end = start + alloc_size;
                // Linux ARM syscall for cacheflush: sys_cacheflush(start, end, flags)
                libc::syscall(0xf0002, start, end, 0);
            }

            ExecutableBuffer {
                ptr: addr as *mut u8,
                size: alloc_size,
            }
        }
    }

    pub unsafe fn execute_loop(&self, limit: u32) -> u64 {
        // Function signature according to AAPCS:
        // Input: r0 = limit
        // Output: r0:r1 (64-bit integer)
        let func: extern "C" fn(u32) -> u64 = std::mem::transmute(self.ptr);
        func(limit)
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}

pub struct Armv7JitEmitter;

impl Armv7JitEmitter {
    /// Generates pure 32-bit ARMv7 Machine Code for:
    /// uint64_t compute_sum(uint32_t limit) {
    ///     uint64_t acc = 0;
    ///     for (uint32_t i = 1; i <= limit; i++) { acc += i; }
    ///     return acc;
    /// }
    pub fn emit_sum_loop() -> Vec<u8> {
        let mut code: Vec<u32> = Vec::new();

        // push {r4, r5, lr}          -> e92d4030
        code.push(0xe92d4030);

        // mov r1, r0                 -> e1a01000 (r1 = limit)
        code.push(0xe1a01000);

        // mov r2, #0                 -> e3a02000 (r2 = acc_low = 0)
        code.push(0xe3a02000);

        // mov r3, #0                 -> e3a03000 (r3 = acc_high = 0)
        code.push(0xe3a03000);

        // mov r4, #1                 -> e3a04001 (r4 = counter i = 1)
        code.push(0xe3a04001);

        // .loop:
        // cmp r4, r1                 -> e1540001 (compare i with limit)
        code.push(0xe1540001);

        // bgt .exit (jump forward 3) -> ca000003
        code.push(0xca000003);

        // adds r2, r2, r4            -> e0922004 (acc_low += i, update carry)
        code.push(0xe0922004);

        // adc r3, r3, #0             -> e2a33000 (acc_high += carry)
        code.push(0xe2a33000);

        // add r4, r4, #1             -> e2844001 (i++)
        code.push(0xe2844001);

        // b .loop (jump back 5)      -> eafffff9
        code.push(0xeafffff9);

        // .exit:
        // mov r0, r2                 -> e1a00002 (r0 = low 32-bit result)
        code.push(0xe1a00002);

        // mov r1, r3                 -> e1a01003 (r1 = high 32-bit result)
        code.push(0xe1a01003);

        // pop {r4, r5, pc}           -> e8bd8030 (return)
        code.push(0xe8bd8030);

        // Convert u32 instructions into Little-Endian bytes
        let mut byte_stream = Vec::with_capacity(code.len() * 4);
        for inst in code {
            byte_stream.extend_from_slice(&inst.to_le_bytes());
        }
        byte_stream
    }
}
