#[repr(C)]
pub struct LumCxxEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LumKVCache {
    _private: [u8; 0],
}

#[link(name = "lum_cxx")]
extern "C" {
    pub fn lum_cxx_engine_new(in_dim: u32, hidden_dim: u32, vocab_size: u32) -> *mut LumCxxEngine;
    pub fn lum_cxx_engine_free(engine: *mut LumCxxEngine);
    pub fn lum_cxx_engine_step_kv(
        engine: *mut LumCxxEngine,
        q_vec: *const i8,
        k_cache_ptr: *const i8,
        seq_len: u32,
        w_ffn: *const i8,
        b_ffn: *const i32,
        out_confidence: *mut u32,
    ) -> u32;

    pub fn lum_kv_cache_new(max_seq: u32, head_dim: u32) -> *mut LumKVCache;
    pub fn lum_kv_cache_free(cache: *mut LumKVCache);
    pub fn lum_kv_cache_append(cache: *mut LumKVCache, k: *const i8, v: *const i8);
    pub fn lum_kv_cache_get_k(cache: *const LumKVCache) -> *const i8;
    pub fn lum_kv_cache_seq_len(cache: *const LumKVCache) -> u32;
    pub fn lum_kv_cache_reset(cache: *mut LumKVCache);
}

pub struct NeuralEngine {
    raw: *mut LumCxxEngine,
}

impl NeuralEngine {
    pub fn new(in_dim: u32, hidden_dim: u32, vocab_size: u32) -> Self {
        let ptr = unsafe { lum_cxx_engine_new(in_dim, hidden_dim, vocab_size) };
        assert!(!ptr.is_null(), "Failed to allocate C++ NeuralEngine");
        Self { raw: ptr }
    }

    pub fn step_kv(
        &self,
        q_vec: &[i8],
        kv_cache: &KVCache,
        w_ffn: &[i8],
        b_ffn: &[i32],
    ) -> (u32, u32) {
        let mut confidence = 0u32;
        let token_id = unsafe {
            lum_cxx_engine_step_kv(
                self.raw,
                q_vec.as_ptr(),
                kv_cache.get_k_ptr(),
                kv_cache.len(),
                w_ffn.as_ptr(),
                b_ffn.as_ptr(),
                &mut confidence as *mut u32,
            )
        };
        (token_id, confidence)
    }
}

impl Drop for NeuralEngine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { lum_cxx_engine_free(self.raw) };
        }
    }
}

pub struct KVCache {
    raw: *mut LumKVCache,
}

impl KVCache {
    pub fn new(max_seq: u32, head_dim: u32) -> Self {
        let ptr = unsafe { lum_kv_cache_new(max_seq, head_dim) };
        assert!(!ptr.is_null(), "Failed to allocate C++ KV-Cache");
        Self { raw: ptr }
    }

    pub fn append(&self, k: &[i8], v: &[i8]) {
        unsafe { lum_kv_cache_append(self.raw, k.as_ptr(), v.as_ptr()) };
    }

    pub fn get_k_ptr(&self) -> *const i8 {
        unsafe { lum_kv_cache_get_k(self.raw) }
    }

    pub fn len(&self) -> u32 {
        unsafe { lum_kv_cache_seq_len(self.raw) }
    }

    pub fn reset(&self) {
        unsafe { lum_kv_cache_reset(self.raw) };
    }
}

impl Drop for KVCache {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { lum_kv_cache_free(self.raw) };
        }
    }
}
