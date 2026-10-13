use std::collections::HashMap;

pub struct ByteLevelBpeTokenizer {
    pub vocab: HashMap<Vec<u8>, u32>,
    pub inverse_vocab: HashMap<u32, Vec<u8>>,
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl ByteLevelBpeTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();

        // 1. Base Vocab: 256 Raw Bytes (0..255)
        for b in 0u8..=255 {
            let byte_slice = vec![b];
            vocab.insert(byte_slice.clone(), b as u32);
            inverse_vocab.insert(b as u32, byte_slice);
        }

        Self {
            vocab,
            inverse_vocab,
            merges: Vec::new(),
        }
    }

    /// Add a BPE merge rule: Pair (A, B) -> Merged Token ID
    pub fn add_merge(&mut self, left: &[u8], right: &[u8]) -> u32 {
        let mut merged = left.to_vec();
        merged.extend_from_slice(right);

        let new_id = self.vocab.len() as u32;
        self.vocab.insert(merged.clone(), new_id);
        self.inverse_vocab.insert(new_id, merged);
        self.merges.push((left.to_vec(), right.to_vec()));
        new_id
    }

    /// Encode input string into Token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let raw_bytes = text.as_bytes();
        let mut tokens: Vec<Vec<u8>> = raw_bytes.iter().map(|&b| vec![b]).collect();

        // Iteratively apply merges in defined order
        for (left, right) in &self.merges {
            let mut i = 0;
            let mut next_tokens = Vec::new();

            while i < tokens.len() {
                if i + 1 < tokens.len() && &tokens[i] == left && &tokens[i + 1] == right {
                    let mut merged = left.clone();
                    merged.extend_from_slice(right);
                    next_tokens.push(merged);
                    i += 2;
                } else {
                    next_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = next_tokens;
        }

        // Map token byte sequences to IDs
        tokens
            .into_iter()
            .map(|tok| *self.vocab.get(&tok).unwrap_or(&0))
            .collect()
    }

    /// Decode Token IDs back to UTF-8 String
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if let Some(tok_bytes) = self.inverse_vocab.get(&id) {
                bytes.extend_from_slice(tok_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

impl Default for ByteLevelBpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}
