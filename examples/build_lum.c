#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Native Vector Struct
typedef struct { double x, y, z; } LumVec3;

static inline LumVec3 vec3_create(double x, double y, double z) {
    return (LumVec3){x, y, z};
}
static inline LumVec3 vec3_sub(LumVec3 a, LumVec3 b) {
    return (LumVec3){a.x - b.x, a.y - b.y, a.z - b.z};
}
static inline double vec3_dot(LumVec3 a, LumVec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static inline double vec3_norm(LumVec3 a) {
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
static inline void vec3_print(LumVec3 a) {
    printf("Vec[%.4f, %.4f, %.4f]\n", a.x, a.y, a.z);
}

int main() {
    LumVec3 u = vec3_create(3.0, 4.0, 0.0);
    LumVec3 v = vec3_create(0.0, 0.0, 5.0);
    printf("%s\n", "--- Native C Luminous Binary Execution ---");
    double distance = pow((3.0*3.0 + 4.0*4.0 + 5.0*5.0), 0.5);
    printf("%.6f\n", (double)(distance));
    return 0;
}