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

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float out;
            memcpy(&out, &bits, sizeof(out));
            return out;
        }

        float v = (float)mant / 1024.0f;
        v = ldexpf(v, -14);
        return sign ? -v : v;
    }

    if (exp == 31) {
        uint32_t bits =
            sign |
            0x7F800000u |
            (mant << 13);

        float out;
        memcpy(&out, &bits, sizeof(out));
        return out;
    }

    uint32_t e =
        exp - 15 + 127;

    uint32_t bits =
        sign |
        (e << 23) |
        (mant << 13);

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/*
 * Exact Q4_K scale/min unpacking corresponding to
 * the verified Rust reference implementation.
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

/*
 * Q4_K block:
 *
 * 0..1    : d
 * 2..3    : dmin
 * 4..15   : scales/mins
 * 16..143 : 128 packed bytes
 *
 * 256 values total.
 *
 * This implementation performs dequantization directly while
 * accumulating into NEON registers. No FP32 weight tensor is
 * created by this kernel.
 */

float luminous_q4k_neon_dot(
    const uint8_t *blocks,
    const float *input,
    size_t n_blocks
) {
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (size_t ib = 0; ib < n_blocks; ++ib) {

        const uint8_t *block = blocks + ib * 144;

        const uint16_t dh =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        const uint16_t dminh =
            (uint16_t)block[2] |
            ((uint16_t)block[3] << 8);

        const float d = half_to_float(dh);
        const float dmin = half_to_float(dminh);

        const uint8_t *q = block + 16;
        const uint8_t *sc = block + 4;
        const float *x = input + ib * 256;

        /*
         * EXACT Q4_K structure:
         *
         * 4 groups
         * each group:
         *   32 packed bytes
         *   -> 32 low-nibble values
         *   -> 32 high-nibble values
         *
         * Scale indices:
         *   group 0: 0,1
         *   group 1: 2,3
         *   group 2: 4,5
         *   group 3: 6,7
         */
        for (int g = 0; g < 4; ++g) {

            uint8_t sc1, mn1;
            uint8_t sc2, mn2;

            get_scale_min_k4(
                g * 2,
                sc,
                &sc1,
                &mn1
            );

            get_scale_min_k4(
                g * 2 + 1,
                sc,
                &sc2,
                &mn2
            );

            const float d1 =
                d * (float)sc1;

            const float m1 =
                dmin * (float)mn1;

            const float d2 =
                d * (float)sc2;

            const float m2 =
                dmin * (float)mn2;

            const uint8_t *qq =
                q + g * 32;

            const float *xx =
                x + g * 64;

            /*
             * LOW NIBBLE -> first 32 values
             */
            for (int l = 0; l < 32; l += 4) {

                uint8_t b0 = qq[l + 0];
                uint8_t b1 = qq[l + 1];
                uint8_t b2 = qq[l + 2];
                uint8_t b3 = qq[l + 3];

                float wv[4];

                wv[0] =
                    d1 * (float)(b0 & 0x0f) - m1;

                wv[1] =
                    d1 * (float)(b1 & 0x0f) - m1;

                wv[2] =
                    d1 * (float)(b2 & 0x0f) - m1;

                wv[3] =
                    d1 * (float)(b3 & 0x0f) - m1;

                float32x4_t w =
                    vld1q_f32(wv);

                float32x4_t xv =
                    vld1q_f32(xx + l);

                acc =
                    vmlaq_f32(
                        acc,
                        w,
                        xv
                    );
            }

            /*
             * HIGH NIBBLE -> second 32 values
             */
            for (int l = 0; l < 32; l += 4) {

                uint8_t b0 = qq[l + 0];
                uint8_t b1 = qq[l + 1];
                uint8_t b2 = qq[l + 2];
                uint8_t b3 = qq[l + 3];

                float wv[4];

                wv[0] =
                    d2 * (float)(b0 >> 4) - m2;

                wv[1] =
                    d2 * (float)(b1 >> 4) - m2;

                wv[2] =
                    d2 * (float)(b2 >> 4) - m2;

                wv[3] =
                    d2 * (float)(b3 >> 4) - m2;

                float32x4_t w =
                    vld1q_f32(wv);

                float32x4_t xv =
                    vld1q_f32(xx + 32 + l);

                acc =
                    vmlaq_f32(
                        acc,
                        w,
                        xv
                    );
            }
        }
    }

    float32x2_t lo =
        vget_low_f32(acc);

    float32x2_t hi =
        vget_high_f32(acc);

    float32x2_t pair =
        vpadd_f32(lo, hi);

    pair =
        vpadd_f32(pair, pair);

    return vget_lane_f32(pair, 0);
}

/*
 * Full Q4_K matrix-vector multiplication.
 *
 * Matrix:
 *     rows × cols
 *
 * Q4_K blocks are stored flattened, so a block may cross
 * a matrix-row boundary when cols is not divisible by 256.
 */

void luminous_q4k_neon_gemv(
    const uint8_t *blocks,
    const float *input,
    float *output,
    size_t rows,
    size_t cols
) {
    const size_t elements = rows * cols;
    const size_t n_blocks = (elements + 255) / 256;

    memset(output, 0, rows * sizeof(float));

    for (size_t ib = 0; ib < n_blocks; ++ib) {

        const uint8_t *block = blocks + ib * 144;

        const uint16_t dh =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        const uint16_t dminh =
            (uint16_t)block[2] |
            ((uint16_t)block[3] << 8);

        const float d = half_to_float(dh);
        const float dmin = half_to_float(dminh);

        const uint8_t *sc = block + 4;
        const uint8_t *q  = block + 16;

        const size_t global_base = ib * 256;

        for (int g = 0; g < 4; ++g) {

            uint8_t sc1, mn1;
            uint8_t sc2, mn2;

            get_scale_min_k4(
                g * 2,
                sc,
                &sc1,
                &mn1
            );

            get_scale_min_k4(
                g * 2 + 1,
                sc,
                &sc2,
                &mn2
            );

            const float d1 = d * (float)sc1;
            const float m1 = dmin * (float)mn1;

            const float d2 = d * (float)sc2;
            const float m2 = dmin * (float)mn2;

            const uint8_t *qq = q + g * 32;

            /*
             * ------------------------------------------------
             * LOW NIBBLES
             * ------------------------------------------------
             */
            int l = 0;

            for (; l <= 28; l += 4) {

                const size_t local =
                    (size_t)g * 64 + (size_t)l;

                const size_t global =
                    global_base + local;

                if (global + 3 >= elements)
                    break;

                const size_t row0 =
                    global / cols;

                const size_t col0 =
                    global % cols;

                /*
                 * Fast vector path only when all 4 values
                 * belong to the same matrix row.
                 */
                if (col0 + 3 < cols) {

                    const uint8_t b0 = qq[l + 0];
                    const uint8_t b1 = qq[l + 1];
                    const uint8_t b2 = qq[l + 2];
                    const uint8_t b3 = qq[l + 3];

                    float wbuf[4] = {
                        d1 * (float)(b0 & 0x0f) - m1,
                        d1 * (float)(b1 & 0x0f) - m1,
                        d1 * (float)(b2 & 0x0f) - m1,
                        d1 * (float)(b3 & 0x0f) - m1
                    };

                    float32x4_t w =
                        vld1q_f32(wbuf);

                    float32x4_t x =
                        vld1q_f32(
                            input + col0
                        );

                    /*
                     * Four independent products.
                     * Reduce ONLY the products.
                     * Then add the resulting scalar once
                     * to the existing row accumulator.
                     */
                    float32x4_t prod =
                        vmulq_f32(w, x);

                    float32x2_t lo =
                        vget_low_f32(prod);

                    float32x2_t hi =
                        vget_high_f32(prod);

                    float32x2_t pair =
                        vpadd_f32(lo, hi);

                    const float sum =
                        vget_lane_f32(pair, 0) +
                        vget_lane_f32(pair, 1);

                    output[row0] += sum;

                } else {
                    for (int k = 0; k < 4; ++k) {
                        const size_t gi =
                            global + (size_t)k;

                        if (gi >= elements)
                            break;

                        const size_t row =
                            gi / cols;

                        const size_t col =
                            gi % cols;

                        const uint8_t b =
                            qq[l + k];

                        const float w =
                            d1 *
                            (float)(b & 0x0f)
                            - m1;

                        output[row] +=
                            w * input[col];
                    }
                }
            }

            /*
             * Remaining low-nibble values.
             */
            for (; l < 32; ++l) {

                const size_t global =
                    global_base +
                    (size_t)g * 64 +
                    (size_t)l;

                if (global >= elements)
                    break;

                const size_t row =
                    global / cols;

                const size_t col =
                    global % cols;

                const float w =
                    d1 *
                    (float)(qq[l] & 0x0f)
                    - m1;

                output[row] +=
                    w * input[col];
            }

            /*
             * ------------------------------------------------
             * HIGH NIBBLES
             * ------------------------------------------------
             */
            l = 0;

            for (; l <= 28; l += 4) {

                const size_t local =
                    (size_t)g * 64 +
                    32 +
                    (size_t)l;

                const size_t global =
                    global_base + local;

                if (global + 3 >= elements)
                    break;

                const size_t row0 =
                    global / cols;

                const size_t col0 =
                    global % cols;

                if (col0 + 3 < cols) {

                    const uint8_t b0 = qq[l + 0];
                    const uint8_t b1 = qq[l + 1];
                    const uint8_t b2 = qq[l + 2];
                    const uint8_t b3 = qq[l + 3];

                    float wbuf[4] = {
                        d2 * (float)(b0 >> 4) - m2,
                        d2 * (float)(b1 >> 4) - m2,
                        d2 * (float)(b2 >> 4) - m2,
                        d2 * (float)(b3 >> 4) - m2
                    };

                    float32x4_t w =
                        vld1q_f32(wbuf);

                    float32x4_t x =
                        vld1q_f32(
                            input + col0
                        );

                    /*
                     * Four independent products need to be
                     * accumulated into the same row.
                     */
                    float32x4_t prod =
                        vmulq_f32(w, x);

                    float32x2_t lo =
                        vget_low_f32(prod);

                    float32x2_t hi =
                        vget_high_f32(prod);

                    float32x2_t pair =
                        vpadd_f32(lo, hi);

                    const float sum =
                        vget_lane_f32(pair, 0) +
                        vget_lane_f32(pair, 1);

                    output[row0] += sum;

                } else {
                    for (int k = 0; k < 4; ++k) {

                        const size_t gi =
                            global + (size_t)k;

                        if (gi >= elements)
                            break;

                        const size_t row =
                            gi / cols;

                        const size_t col =
                            gi % cols;

                        const uint8_t b =
                            qq[l + k];

                        const float w =
                            d2 *
                            (float)(b >> 4)
                            - m2;

                        output[row] +=
                            w * input[col];
                    }
                }
            }

            for (; l < 32; ++l) {

                const size_t global =
                    global_base +
                    (size_t)g * 64 +
                    32 +
                    (size_t)l;

                if (global >= elements)
                    break;

                const size_t row =
                    global / cols;

                const size_t col =
                    global % cols;

                const float w =
                    d2 *
                    (float)(qq[l] >> 4)
                    - m2;

                output[row] +=
                    w * input[col];
            }
        }
    }
}
