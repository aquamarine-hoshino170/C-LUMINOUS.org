.syntax unified
.arm
.text

.global lum_bigint_add_n
.type lum_bigint_add_n, %function

@ -----------------------------------------------------------------------------
@ lum_bigint_add_n(r0 = *mut u32 res, r1 = *const u32 a, r2 = *const u32 b, r3 = limbs) -> carry (r0)
@ -----------------------------------------------------------------------------
lum_bigint_add_n:
    push    {r4-r8, lr}
    movs    r4, #0          @ Clear Carry Flag initially
.loop_add:
    cmp     r3, #0
    beq     .done_add
    ldr     r5, [r1], #4    @ Load limb from A
    ldr     r6, [r2], #4    @ Load limb from B
    
    @ Restore Carry, Add with Carry, Preserve Carry
    msr     APSR_nzcvq, r4
    adcs    r5, r5, r6
    mrs     r4, APSR_nzcvq
    
    str     r5, [r0], #4    @ Store into Res
    subs    r3, r3, #1
    b       .loop_add
.done_add:
    msr     APSR_nzcvq, r4
    mov     r0, #0
    adc     r0, r0, #0      @ Return final overflow carry
    pop     {r4-r8, pc}
.size lum_bigint_add_n, .-lum_bigint_add_n


.global lum_bigint_sub_n
.type lum_bigint_sub_n, %function

@ -----------------------------------------------------------------------------
@ lum_bigint_sub_n(r0 = *mut u32 res, r1 = *const u32 a, r2 = *const u32 b, r3 = limbs) -> borrow (r0)
@ -----------------------------------------------------------------------------
lum_bigint_sub_n:
    push    {r4-r8, lr}
    mov     r4, #0x20000000 @ Set Carry Flag (No borrow initially in ARM)
.loop_sub:
    cmp     r3, #0
    beq     .done_sub
    ldr     r5, [r1], #4
    ldr     r6, [r2], #4
    
    msr     APSR_nzcvq, r4
    sbcs    r5, r5, r6      @ Subtract with Borrow
    mrs     r4, APSR_nzcvq
    
    str     r5, [r0], #4
    subs    r3, r3, #1
    b       .loop_sub
.done_sub:
    msr     APSR_nzcvq, r4
    mov     r0, #0
    adc     r0, r0, #0      @ 0 if borrowed, 1 if clean
    eor     r0, r0, #1      @ Return 1 if borrow occurred, else 0
    pop     {r4-r8, pc}
.size lum_bigint_sub_n, .-lum_bigint_sub_n


.global lum_bigint_mul_acc
.type lum_bigint_mul_acc, %function

@ -----------------------------------------------------------------------------
@ lum_bigint_mul_acc(r0 = *mut u32 res_lane, r1 = *const u32 a, r2 = scalar_b, r3 = a_limbs)
@ -----------------------------------------------------------------------------
lum_bigint_mul_acc:
    push    {r4-r10, lr}
    mov     r4, #0          @ High carry = 0
.loop_mul:
    cmp     r3, #0
    beq     .done_mul
    ldr     r5, [r1], #4    @ a[i]
    ldr     r6, [r0]        @ res[i+j]
    
    @ 64-bit multiply: r8:r7 = a[i] * scalar_b
    umull   r7, r8, r5, r2
    
    @ Add previous limb accumulation and carry
    adds    r7, r7, r6
    adc     r8, r8, #0
    adds    r7, r7, r4
    adc     r8, r8, #0
    
    str     r7, [r0], #4
    mov     r4, r8          @ Carry forward upper 32-bit
    subs    r3, r3, #1
    b       .loop_mul
.done_mul:
    @ Store final carry to the next lane
    ldr     r6, [r0]
    add     r6, r6, r4
    str     r6, [r0]
    pop     {r4-r10, pc}
.size lum_bigint_mul_acc, .-lum_bigint_mul_acc
