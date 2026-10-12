#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

static float half_to_float(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;

    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            float v = (float)mant / 1024.0f;
            v = ldexpf(v, -14);
            return sign ? -v : v;
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t e = exp - 15 + 127;
        out = sign | (e << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &out, sizeof(result));
    return result;
}

/*
 * Q4_K block:
 *
 * bytes 0..1    : d
 * bytes 2..3    : dmin
 * bytes 4..15   : scales/mins
 * bytes 16..143  : 4-bit quantized values
 *
 * 256 values/block.
 */
static void get_scale_min_k4(
    size_t j,
    const uint8_t *q,
    uint8_t *scale,
    uint8_t *min
) {
    if (j < 4) {
        *scale = q[j] & 63u;
        *min   = q[j + 4] & 63u;
    } else {
        *scale =
            (q[j + 4] & 0x0Fu) |
            ((q[j - 4] >> 6) << 4);

        *min =
            (q[j + 4] >> 4) |
            ((q[j] >> 6) << 4);
    }
}

float luminous_q4k_neon_dot(
    const uint8_t *blocks,
    const float *input,
    size_t n_blocks
) {
    float32x4_t vacc = vdupq_n_f32(0.0f);

    size_t input_pos = 0;

    for (size_t block_index = 0;
         block_index < n_blocks;
         ++block_index) {

        const uint8_t *block =
            blocks + block_index * 144;

        uint16_t dh =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        uint16_t dminh =
            (uint16_t)block[2] |
            ((uint16_t)block[3] << 8);

        float d = half_to_float(dh);
        float dmin = half_to_float(dminh);

        const uint8_t *scales = block + 4;
        const uint8_t *qs     = block + 16;

        /*
         * Four groups × 64 values.
         *
         * Each group:
         *   32 low nibbles
         *   32 high nibbles
         */
        for (size_t group = 0; group < 4; ++group) {

            size_t qbase = group * 32;
            size_t xbase = input_pos + group * 64;

            uint8_t sc0, mn0;
            uint8_t sc1, mn1;

            get_scale_min_k4(
                group * 2,
                scales,
                &sc0,
                &mn0
            );

            get_scale_min_k4(
                group * 2 + 1,
                scales,
                &sc1,
                &mn1
            );

            float scale0 = d * (float)sc0;
            float min0   = dmin * (float)mn0;

            float scale1 = d * (float)sc1;
            float min1   = dmin * (float)mn1;

            /*
             * First 32 values.
             */
            for (size_t j = 0; j < 32; j += 4) {

                uint8_t packed =
                    qs[qbase + j / 2];

                uint8_t q0 = packed & 0x0F;
                uint8_t q1 = packed >> 4;

                /*
                 * Four values are packed across two bytes.
                 */
                uint8_t packed2 =
                    qs[qbase + j / 2 + 1];

                float32x4_t w;

                float tmp[4];

                tmp[0] =
                    scale0 * (float)q0 - min0;

                tmp[1] =
                    scale0 * (float)q1 - min0;

                tmp[2] =
                    scale0 * (float)(packed2 & 0x0F) - min0;

                tmp[3] =
                    scale0 * (float)(packed2 >> 4) - min0;

                w = vld1q_f32(tmp);

                float32x4_t x =
                    vld1q_f32(input + xbase + j);

                vacc = vmlaq_f32(vacc, w, x);
            }

            /*
             * Second 32 values.
             *
             * Q4_K stores the upper nibbles as the second
             * half of the 64-value group.
             */
            for (size_t j = 0; j < 32; j += 4) {

                size_t byte_index = qbase + j / 2;

                uint8_t b0 = qs[byte_index];
                uint8_t b1 = qs[byte_index + 1];

                float tmp[4];

                tmp[0] =
                    scale1 * (float)(b0 >> 4) - min1;

                tmp[1] =
                    scale1 * (float)(b1 >> 4) - min1;

                uint8_t b2 = qs[byte_index + 2];

                tmp[2] =
                    scale1 * (float)(b2 >> 4) - min1;

                uint8_t b3 = qs[byte_index + 3];

                tmp[3] =
                    scale1 * (float)(b3 >> 4) - min1;

                float32x4_t w =
                    vld1q_f32(tmp);

                float32x4_t x =
                    vld1q_f32(
                        input + xbase + 32 + j
                    );

                vacc = vmlaq_f32(vacc, w, x);
            }
        }

        input_pos += 256;
    }

    float32x2_t lo =
        vget_low_f32(vacc);

    float32x2_t hi =
        vget_high_f32(vacc);

    float32x2_t pair =
        vpadd_f32(lo, hi);

    pair =
        vpadd_f32(pair, pair);

    return vget_lane_f32(pair, 0);
}
