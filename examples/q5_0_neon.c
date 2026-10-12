#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x03FFu;
    uint32_t out;

    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            /*
             * Normalize subnormal.
             */
            uint32_t f = frac;
            int e = -14;

            while ((f & 0x0400u) == 0) {
                f <<= 1;
                e--;
            }

            f &= 0x03FFu;

            uint32_t e32 = (uint32_t)(e + 127);
            out = sign | (e32 << 23) | (f << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        uint32_t e32 = exp - 15 + 127;
        out = sign | (e32 << 23) | (frac << 13);
    }

    float result;
    memcpy(&result, &out, sizeof(result));
    return result;
}

/*
 * GGML Q5_0 block:
 *
 *   d   : fp16          [0..2)
 *   qh  : uint32        [2..6)
 *   qs  : uint8[16]     [6..22)
 *
 * 32 weights per block.
 *
 * ql = 4-bit low part
 * qh = 1-bit high part
 *
 * value = d * (q5 - 16)
 */
static void q5_0_decode_block(
    const uint8_t *block,
    float *out
)
{
    uint16_t d_bits =
        (uint16_t)block[0] |
        ((uint16_t)block[1] << 8);

    const float d =
        half_to_float(d_bits);

    uint32_t qh =
        (uint32_t)block[2] |
        ((uint32_t)block[3] << 8) |
        ((uint32_t)block[4] << 16) |
        ((uint32_t)block[5] << 24);

    const uint8_t *qs =
        block + 6;

    for (int j = 0; j < 16; ++j) {
        const uint8_t q = qs[j];

        int q0 =
            (q & 0x0F) |
            (((qh >> j) & 1u) << 4);

        int q1 =
            (q >> 4) |
            (((qh >> (j + 16)) & 1u) << 4);

        out[j] =
            d * (float)(q0 - 16);

        out[j + 16] =
            d * (float)(q1 - 16);
    }
}

static float hsum4(float32x4_t v)
{
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);

    float32x2_t p = vpadd_f32(lo, hi);

    return
        vget_lane_f32(p, 0) +
        vget_lane_f32(p, 1);
}

/*
 * Q5_0 GEMV.
 *
 * Matrix is stored flattened in GGUF row-major logical layout:
 *
 *   rows × cols
 *
 * Q5_0 blocks contain 32 consecutive values.
 *
 * NEON fast path is used only when four values stay
 * inside the same output row.
 */


void luminous_q5_0_debug_block(
    const uint8_t *block,
    float *out32
)
{
    uint16_t d_bits =
        (uint16_t)block[0] |
        ((uint16_t)block[1] << 8);

    const float d = half_to_float(d_bits);

    uint32_t qh =
        (uint32_t)block[2] |
        ((uint32_t)block[3] << 8) |
        ((uint32_t)block[4] << 16) |
        ((uint32_t)block[5] << 24);

    const uint8_t *qs = block + 6;

    for (int j = 0; j < 16; ++j) {
        int q0 =
            (qs[j] & 0x0F) |
            (((qh >> j) & 1u) << 4);

        int q1 =
            (qs[j] >> 4) |
            (((qh >> (j + 16)) & 1u) << 4);

        out32[j] =
            d * (float)(q0 - 16);

        out32[j + 16] =
            d * (float)(q1 - 16);
    }
}


float luminous_q5_0_debug_dot32(
    const uint8_t *block,
    const float *input
)
{
    uint16_t d_bits =
        (uint16_t)block[0] |
        ((uint16_t)block[1] << 8);

    const float d = half_to_float(d_bits);

    uint32_t qh =
        (uint32_t)block[2] |
        ((uint32_t)block[3] << 8) |
        ((uint32_t)block[4] << 16) |
        ((uint32_t)block[5] << 24);

    const uint8_t *qs = block + 6;

    float acc = 0.0f;

    for (int j = 0; j < 32; ++j) {
        int q;

        if (j < 16) {
            q =
                (qs[j] & 0x0F) |
                (((qh >> j) & 1u) << 4);
        } else {
            const int k = j - 16;

            q =
                (qs[k] >> 4) |
                (((qh >> (k + 16)) & 1u) << 4);
        }

        const float w =
            d * (float)(q - 16);

        acc += w * input[j];
    }

    return acc;
}


void luminous_q5_0_debug_row_blocks(
    const uint8_t *data,
    size_t cols,
    const float *input,
    float *out_blocks
)
{
    const size_t blocks_per_row = cols / 32;

    for (size_t block_in_row = 0;
         block_in_row < blocks_per_row;
         ++block_in_row) {

        const size_t block_index =
            block_in_row;

        const uint8_t *block =
            data + block_index * 22;

        uint16_t d_bits =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        const float d =
            half_to_float(d_bits);

        uint32_t qh =
            (uint32_t)block[2] |
            ((uint32_t)block[3] << 8) |
            ((uint32_t)block[4] << 16) |
            ((uint32_t)block[5] << 24);

        const uint8_t *qs = block + 6;

        float acc = 0.0f;

        for (int j = 0; j < 32; ++j) {

            int q;

            if (j < 16) {
                q =
                    (qs[j] & 0x0F) |
                    (((qh >> j) & 1u) << 4);
            } else {
                const int k = j - 16;

                q =
                    (qs[k] >> 4) |
                    (((qh >> (k + 16)) & 1u) << 4);
            }

            const float w =
                d * (float)(q - 16);

            acc +=
                w * input[block_in_row * 32 + (size_t)j];
        }

        out_blocks[block_in_row] = acc;
    }
}

void luminous_q5_0_gemv_scalar(
    const uint8_t *data,
    size_t rows,
    size_t cols,
    const float *input,
    float *output
)
{
    const size_t blocks_per_row = cols / 32;

    memset(output, 0, rows * sizeof(float));

    /*
     * GGUF/GGML logical layout:
     *
     *   [896,128]
     *
     * means:
     *   128 output rows
     *   896 input values per row
     *
     * Each row therefore contains:
     *
     *   896 / 32 = 28 Q5_0 blocks.
     *
     * Block order is explicitly:
     *
     *   row0 block0..27
     *   row1 block0..27
     *   ...
     */
    for (size_t row = 0; row < rows; ++row) {

        float acc = 0.0f;

        for (size_t block_in_row = 0;
             block_in_row < blocks_per_row;
             ++block_in_row) {

            const size_t block_index =
                row * blocks_per_row + block_in_row;

            const uint8_t *block =
                data + block_index * 22;

            uint16_t d_bits =
                (uint16_t)block[0] |
                ((uint16_t)block[1] << 8);

            const float d =
                half_to_float(d_bits);

            uint32_t qh =
                (uint32_t)block[2] |
                ((uint32_t)block[3] << 8) |
                ((uint32_t)block[4] << 16) |
                ((uint32_t)block[5] << 24);

            const uint8_t *qs =
                block + 6;

            const size_t input_base =
                block_in_row * 32;

            for (int j = 0; j < 32; ++j) {

                int q;

                if (j < 16) {
                    q =
                        (qs[j] & 0x0F) |
                        (((qh >> j) & 1u) << 4);
                } else {
                    const int k = j - 16;

                    q =
                        (qs[k] >> 4) |
                        (((qh >> (k + 16)) & 1u) << 4);
                }

                const float weight =
                    d * (float)(q - 16);

                acc +=
                    weight *
                    input[input_base + (size_t)j];
            }
        }

        output[row] = acc;
    }
}

void luminous_q5_0_gemv(
    const uint8_t *data,
    size_t rows,
    size_t cols,
    const float *input,
    float *output
)
{
    const size_t blocks_per_row = cols / 32;

    memset(output, 0, rows * sizeof(float));

    for (size_t row = 0; row < rows; ++row) {

        float acc_row = 0.0f;

        for (size_t block_in_row = 0;
             block_in_row < blocks_per_row;
             ++block_in_row) {

            const size_t block_index =
                row * blocks_per_row + block_in_row;

            const uint8_t *block =
                data + block_index * 22;

            uint16_t d_bits =
                (uint16_t)block[0] |
                ((uint16_t)block[1] << 8);

            const float d =
                half_to_float(d_bits);

            uint32_t qh =
                (uint32_t)block[2] |
                ((uint32_t)block[3] << 8) |
                ((uint32_t)block[4] << 16) |
                ((uint32_t)block[5] << 24);

            const uint8_t *qs =
                block + 6;

            const size_t input_base =
                block_in_row * 32;

            for (int j = 0; j < 32; j += 4) {

                float wbuf[4];

                for (int k = 0; k < 4; ++k) {

                    const int idx = j + k;

                    int q;

                    if (idx < 16) {
                        q =
                            (qs[idx] & 0x0F) |
                            (((qh >> idx) & 1u) << 4);
                    } else {
                        const int s2 = idx - 16;

                        q =
                            (qs[s2] >> 4) |
                            (((qh >> (s2 + 16)) & 1u) << 4);
                    }

                    wbuf[k] =
                        d * (float)(q - 16);
                }

                float32x4_t w =
                    vld1q_f32(wbuf);

                float32x4_t x =
                    vld1q_f32(input + input_base + (size_t)j);

                float32x4_t prod =
                    vmulq_f32(w, x);

                acc_row += hsum4(prod);
            }
        }

        output[row] = acc_row;
    }
}

