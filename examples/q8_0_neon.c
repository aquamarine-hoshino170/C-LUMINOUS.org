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
            uint32_t f = frac;
            int e = -14;

            while ((f & 0x0400u) == 0) {
                f <<= 1;
                e--;
            }

            f &= 0x03FFu;
            out =
                sign |
                ((uint32_t)(e + 127) << 23) |
                (f << 13);
        }
    } else if (exp == 31) {
        out =
            sign |
            0x7F800000u |
            (frac << 13);
    } else {
        out =
            sign |
            ((uint32_t)(exp - 15 + 127) << 23) |
            (frac << 13);
    }

    float result;
    memcpy(&result, &out, sizeof(result));
    return result;
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
 * GGML Q8_0:
 *
 * block:
 *   d  : fp16
 *   qs : int8[32]
 *
 * bytes = 2 + 32 = 34
 *
 * value = d * qs[i]
 */
void luminous_q8_0_gemv_scalar(
    const uint8_t *data,
    size_t rows,
    size_t cols,
    const float *input,
    float *output
)
{
    const size_t blocks =
        rows * cols / 32;

    memset(output, 0, rows * sizeof(float));

    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t *block =
            data + b * 34;

        uint16_t d_bits =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        const float d =
            half_to_float(d_bits);

        const int8_t *qs =
            (const int8_t *)(block + 2);

        const size_t base =
            b * 32;

        for (int j = 0; j < 32; ++j) {
            const size_t global =
                base + (size_t)j;

            const size_t row =
                global / cols;

            const size_t col =
                global % cols;

            output[row] +=
                d *
                (float)qs[j] *
                input[col];
        }
    }
}

void luminous_q8_0_gemv(
    const uint8_t *data,
    size_t rows,
    size_t cols,
    const float *input,
    float *output
)
{
    const size_t blocks =
        rows * cols / 32;

    memset(output, 0, rows * sizeof(float));

    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t *block =
            data + b * 34;

        uint16_t d_bits =
            (uint16_t)block[0] |
            ((uint16_t)block[1] << 8);

        const float d =
            half_to_float(d_bits);

        const int8_t *qs =
            (const int8_t *)(block + 2);

        const size_t base =
            b * 32;

        /*
         * 896 and 128 are both divisible by 32,
         * so each Q8_0 block stays inside a row.
         */
        const size_t global =
            base;

        const size_t row =
            global / cols;

        const size_t col =
            global % cols;

        /*
         * Four int8 values → four float weights.
         */
        for (int j = 0; j < 32; j += 8) {
            int16x8_t q16 =
                vmovl_s8(
                    vld1_s8(qs + j)
                );

            int32x4_t qlo =
                vmovl_s16(
                    vget_low_s16(q16)
                );

            int32x4_t qhi =
                vmovl_s16(
                    vget_high_s16(q16)
                );

            float32x4_t qf0 =
                vcvtq_f32_s32(qlo);

            float32x4_t qf1 =
                vcvtq_f32_s32(qhi);

            const float32x4_t df =
                vdupq_n_f32(d);

            qf0 =
                vmulq_f32(qf0, df);

            qf1 =
                vmulq_f32(qf1, df);

            float32x4_t x0 =
                vld1q_f32(input + col + j);

            float32x4_t x1 =
                vld1q_f32(input + col + j + 4);

            float32x4_t p0 =
                vmulq_f32(qf0, x0);

            float32x4_t p1 =
                vmulq_f32(qf1, x1);

            output[row] +=
                hsum4(p0) +
                hsum4(p1);
        }
    }
}
