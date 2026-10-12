use lum_rust::gguf_loader::q4k_reference::{
    dequantize_block, get_scale_min_k4, half_to_f32, BLOCK_BYTES, BLOCK_SIZE,
};

#[test]
fn q4k_constants() {
    assert_eq!(BLOCK_SIZE, 256);
    assert_eq!(BLOCK_BYTES, 144);
}

#[test]
fn half_values() {
    assert_eq!(half_to_f32(0x0000), 0.0);
    assert_eq!(half_to_f32(0x3c00), 1.0);
    assert_eq!(half_to_f32(0xc000), -2.0);
}

#[test]
fn scale_min_unpacking() {
    // First four scale/min pairs are stored directly.
    let mut q = [0u8; 12];

    q[0] = 1;
    q[1] = 2;
    q[2] = 3;
    q[3] = 4;

    q[4] = 5;
    q[5] = 6;
    q[6] = 7;
    q[7] = 8;

    assert_eq!(get_scale_min_k4(0, &q), (1, 5));
    assert_eq!(get_scale_min_k4(1, &q), (2, 6));
    assert_eq!(get_scale_min_k4(2, &q), (3, 7));
    assert_eq!(get_scale_min_k4(3, &q), (4, 8));
}

#[test]
fn deterministic_q4k_vector() {
    let mut block = [0u8; BLOCK_BYTES];

    // d = 1.0
    block[0] = 0x00;
    block[1] = 0x3c;

    // dmin = 0.0
    block[2] = 0x00;
    block[3] = 0x00;

    // Scale groups:
    // group 0 -> scale 1, min 0
    // group 1 -> scale 2, min 0
    // group 2 -> scale 3, min 0
    // group 3 -> scale 4, min 0
    block[4] = 1;
    block[5] = 2;
    block[6] = 3;
    block[7] = 4;

    // mins for groups 0..3 remain zero.
    //
    // For groups 4..7 we use simple low values.
    block[8] = 5;
    block[9] = 6;
    block[10] = 7;
    block[11] = 8;

    // q = 0x21 => low nibble = 1, high nibble = 2.
    // First 64 values therefore use group 0 and group 1.
    for i in 0..32 {
        block[16 + i] = 0x21;
    }

    let values = dequantize_block(&block).expect("valid Q4_K block");

    assert_eq!(values.len(), 256);

    // group 0: 1 * 1
    assert_eq!(values[0], 1.0);

    // group 1: 2 * 2
    assert_eq!(values[32], 4.0);

    // Every value must be finite.
    assert!(values.iter().all(|x| x.is_finite()));
}

#[test]
fn q4k_zero_block() {
    let block = [0u8; BLOCK_BYTES];

    let values = dequantize_block(&block).expect("zero Q4_K block");

    assert_eq!(values.len(), BLOCK_SIZE);

    for value in values {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn q4k_rejects_wrong_size() {
    let block = [0u8; 100];

    assert!(dequantize_block(&block).is_err());
}
