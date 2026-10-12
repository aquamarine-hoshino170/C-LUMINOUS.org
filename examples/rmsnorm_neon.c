#include <arm_neon.h>
#include <stddef.h>
#include <math.h>

/*
 * ARMv7 NEON RMSNorm
 *
 * y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 */

void luminous_rmsnorm_neon(
    const float *input,
    const float *weight,
    float *output,
    size_t n,
    float eps
) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    size_t i = 0;

    /*
     * Accumulate squares in two independent accumulators
     * to reduce dependency chains.
     */
    for (; i + 8 <= n; i += 8) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);

        sum0 = vmlaq_f32(sum0, x0, x0);
        sum1 = vmlaq_f32(sum1, x1, x1);
    }

    float32x4_t total = vaddq_f32(sum0, sum1);

    float32x2_t lo = vget_low_f32(total);
    float32x2_t hi = vget_high_f32(total);
    float32x2_t pair = vpadd_f32(lo, hi);

    float sum_sq =
        vget_lane_f32(pair, 0) +
        vget_lane_f32(pair, 1);

    /*
     * Scalar tail.
     */
    for (; i < n; ++i) {
        sum_sq += input[i] * input[i];
    }

    float mean_sq = sum_sq / (float)n;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    float32x4_t inv = vdupq_n_f32(inv_rms);

    i = 0;

    /*
     * Vectorized normalization + scale.
     */
    for (; i + 4 <= n; i += 4) {
        float32x4_t x =
            vld1q_f32(input + i);

        float32x4_t w =
            vld1q_f32(weight + i);

        float32x4_t y =
            vmulq_f32(x, inv);

        y =
            vmulq_f32(y, w);

        vst1q_f32(output + i, y);
    }

    for (; i < n; ++i) {
        output[i] =
            input[i] * inv_rms * weight[i];
    }
}
