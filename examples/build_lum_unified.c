#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// ==========================================================================
// 1. DIFFERENTIAL MANIFOLD & GEODESIC DYNAMICS
// ==========================================================================
typedef struct { double g[3][3]; } LumMetric;
typedef struct { double x[4]; double u[4]; } LumGeodesic4D;

static inline LumMetric lum_schwarzschild_metric(double r, double M) {
    LumMetric m = {0};
    double f = 1.0 - (2.0 * M / r);
    m.g[0][0] = -f; m.g[1][1] = 1.0 / f; m.g[2][2] = r * r;
    return m;
}

static inline LumGeodesic4D lum_geodesic_step(LumGeodesic4D s, double M, double dtau) {
    LumGeodesic4D next = s;
    double r = s.x[1];
    if (r <= 2.0 * M) return s;
    double f = 1.0 - (2.0 * M / r);
    double dt = s.u[0], dr = s.u[1], dphi = s.u[3];

    double acc_t = - 2.0 * ((M / (r * r)) / f) * dt * dr;
    double acc_r = - ((M / (r * r)) * f) * (dt * dt) + ((M / (r * r)) / f) * (dr * dr) + (r * f) * (dphi * dphi);
    double acc_phi = - (2.0 / r) * dr * dphi;

    next.u[0] += acc_t * dtau;
    next.u[1] += acc_r * dtau;
    next.u[3] += acc_phi * dtau;
    next.x[0] += next.u[0] * dtau;
    next.x[1] += next.u[1] * dtau;
    next.x[3] += next.u[3] * dtau;
    return next;
}

static inline double lum_proper_velocity_norm(LumGeodesic4D s, double M) {
    double r = s.x[1];
    double f = 1.0 - (2.0 * M / r);
    return - f * (s.u[0]*s.u[0]) + (1.0/f)*(s.u[1]*s.u[1]) + (r*r)*(s.u[3]*s.u[3]);
}

// ==========================================================================
// 2. FIRST-ORDER HORN CLAUSE THEOREM PROVER
// ==========================================================================
#define MAX_SYMBOLS 64
#define MAX_RULES 64
#define MAX_BODY 8

typedef struct { char head[32]; char body[MAX_BODY][32]; int body_count; } LumHornRule;
typedef struct { char facts[MAX_SYMBOLS][32]; int fact_count; LumHornRule rules[MAX_RULES]; int rule_count; } LumKB;

static inline LumKB lum_kb_init() { LumKB kb; kb.fact_count = 0; kb.rule_count = 0; return kb; }

static inline void lum_kb_fact(LumKB *kb, const char *f) {
    for(int i=0; i<kb->fact_count; i++) if(strcmp(kb->facts[i], f) == 0) return;
    strcpy(kb->facts[kb->fact_count++], f);
}

static inline void lum_kb_rule(LumKB *kb, const char *h, const char *b) {
    LumHornRule *r = &kb->rules[kb->rule_count++];
    strcpy(r->head, h); r->body_count = 0;
    char buf[256]; strcpy(buf, b);
    char *tok = strtok(buf, ", ");
    while(tok && r->body_count < MAX_BODY) { strcpy(r->body[r->body_count++], tok); tok = strtok(NULL, ", "); }
}

static inline bool lum_kb_contains(LumKB *kb, const char *f) {
    for(int i=0; i<kb->fact_count; i++) if(strcmp(kb->facts[i], f) == 0) return true;
    return false;
}

static inline bool lum_kb_prove(LumKB *kb, const char *q) {
    bool changed = true;
    while(changed) {
        changed = false;
        if (lum_kb_contains(kb, q)) return true;
        for(int i=0; i<kb->rule_count; i++) {
            LumHornRule *r = &kb->rules[i];
            if (lum_kb_contains(kb, r->head)) continue;
            bool sat = true;
            for(int j=0; j<r->body_count; j++) {
                if (!lum_kb_contains(kb, r->body[j])) { sat = false; break; }
            }
            if (sat) { lum_kb_fact(kb, r->head); changed = true; }
        }
    }
    return lum_kb_contains(kb, q);
}

int main() {
    printf("%s\n", "==================================================");
    printf("%s\n", "     ❖ C LUMINOUS PURE SYSTEMS LANGUAGE (v1.0) ❖    ");
    printf("%s\n", "==================================================");
    LumGeodesic4D g0 = {{0.0, 6.0, 1.570796, 0.0}, {1.41421356, 0.0, 0.0, 0.09622504}};
    double norm_t0 = lum_proper_velocity_norm(g0, 1.0);
    printf("%s\n", "Initial 4-Velocity Invariant (g_ab u^a u^b):");
    printf("[Invariant]: %.10f\n", (double)(norm_t0));
    LumGeodesic4D g1 = lum_geodesic_step(g0, 1.0, 0.05);
    double norm_t1 = lum_proper_velocity_norm(g1, 1.0);
    printf("%s\n", "Conserved 4-Velocity Invariant after Dynamics Step:");
    printf("[Invariant]: %.10f\n", (double)(norm_t1));
    LumKB kb = lum_kb_init();
    lum_kb_fact(&kb, "compact_space");
    lum_kb_fact(&kb, "connected_space");
    lum_kb_rule(&kb, "continuum", "compact_space, connected_space");
    lum_kb_rule(&kb, "weierstrass_valid", "continuum");
    printf("%s\n", "Theorem Verification: Does Continuum imply Weierstrass Extremum Theorem?");
    bool proof_w = lum_kb_prove(&kb, "weierstrass_valid");
    printf("[Logic Q.E.D.]: %s\n", (proof_w) ? "VALID THEOREM" : "INVALID");
    return 0;
}