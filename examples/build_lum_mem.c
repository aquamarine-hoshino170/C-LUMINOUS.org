#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// ==========================================================================
// DIRECT MEMORY POINTER & DYNAMIC BUFFER RUNTIME
// ==========================================================================
typedef struct {
    double *data;
    size_t capacity;
} LumPtr;

static inline LumPtr lum_alloc_ptr(size_t count) {
    LumPtr p;
    p.data = (double *)malloc(count * sizeof(double));
    p.capacity = count;
    memset(p.data, 0, count * sizeof(double));
    return p;
}

static inline void lum_free_ptr(LumPtr *p) {
    if (p->data) {
        free(p->data);
        p->data = NULL;
        p->capacity = 0;
    }
}

static inline void lum_ptr_set(LumPtr *p, size_t offset, double val) {
    if (offset < p->capacity) {
        p->data[offset] = val;
    }
}

static inline double lum_ptr_get(LumPtr *p, size_t offset) {
    if (offset < p->capacity) {
        return p->data[offset];
    }
    return 0.0;
}

static inline double lum_ptr_sum(LumPtr *p) {
    double total = 0.0;
    for (size_t i = 0; i < p->capacity; i++) {
        total += p->data[i];
    }
    return total;
}

int main() {
    printf("%s\n", "=== DIRECT MEMORY POINTER & DYNAMIC HEAP ALLOCATION ===");
    LumPtr buffer = lum_alloc_ptr(4);
    lum_ptr_set(&buffer, 0, 1.050000);
    lum_ptr_set(&buffer, 1, 2.718281);
    lum_ptr_set(&buffer, 2, 3.141592);
    lum_ptr_set(&buffer, 3, 0.577215);
    printf("%s\n", "--- Reading Value at Pointer Offset 2 ---");
    double val2 = lum_ptr_get(&buffer, 2);
    printf("[Memory Value]: %.6f\n", (double)(val2));
    printf("%s\n", "--- Sum of All Heap-Allocated Elements ---");
    double total_sum = lum_ptr_sum(&buffer);
    printf("[Memory Value]: %.6f\n", (double)(total_sum));
    lum_free_ptr(&buffer);
    printf("%s\n", "Memory Successfully Deallocated with Zero Leaks.");
    return 0;
}