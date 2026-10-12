#include <arm_neon.h>
#include <stddef.h>

float luminous_neon_dot(
    const float *a,
    const float *b,
    size_t n
) {
    float32x4_t acc = vdupq_n_f32(0.0f);

    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        acc = vmlaq_f32(acc, va, vb);
    }

    float32x2_t low  = vget_low_f32(acc);
    float32x2_t high = vget_high_f32(acc);

    float32x2_t pair = vpadd_f32(low, high);
    pair = vpadd_f32(pair, pair);

    float result = vget_lane_f32(pair, 0);

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
