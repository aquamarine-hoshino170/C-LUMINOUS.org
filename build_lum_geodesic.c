#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// ==========================================================================
// 4-VECTOR MANIFOLD & GEODESIC INTEGRATOR
// ==========================================================================
typedef struct {
    double x[4]; // Coordinates: t, r, theta, phi
    double u[4]; // Four-velocity: dt/dtau, dr/dtau, dtheta/dtau, dphi/dtau
} GeodesicState4D;

// Schwarzschild Geodesic Acceleration: a^mu = - Gamma^mu_ab * u^a * u^b
static inline GeodesicState4D geodesic_step(GeodesicState4D s, double M, double dtau) {
    GeodesicState4D next = s;
    double r = s.x[1];
    if (r <= 2.0 * M) return s; // Horizon boundary condition

    double f = 1.0 - (2.0 * M / r);
    double dt = s.u[0];
    double dr = s.u[1];
    double dphi = s.u[3];

    // Christoffel acceleration components for equatorial plane (theta = pi/2, dtheta = 0)
    // d^2 t / dtau^2 = - 2 * Gamma^t_tr * dt * dr
    double acc_t = - 2.0 * ((M / (r * r)) / f) * dt * dr;

    // d^2 r / dtau^2 = - Gamma^r_tt * dt^2 - Gamma^r_rr * dr^2 - Gamma^r_phiphi * dphi^2
    // where Gamma^r_phiphi = - r * f
    double acc_r = - ((M / (r * r)) * f) * (dt * dt) \
                   + ((M / (r * r)) / f) * (dr * dr) \
                   + (r * f) * (dphi * dphi);

    // d^2 phi / dtau^2 = - 2 * Gamma^phi_rphi * dr * dphi = - (2/r) * dr * dphi
    double acc_phi = - (2.0 / r) * dr * dphi;

    // Symplectic-Euler update
    next.u[0] += acc_t * dtau;
    next.u[1] += acc_r * dtau;
    next.u[3] += acc_phi * dtau;

    next.x[0] += next.u[0] * dtau;
    next.x[1] += next.u[1] * dtau;
    next.x[3] += next.u[3] * dtau;

    return next;
}

// 4-Velocity Normalization Invariant: g_mu_nu * u^mu * u^nu = -1.0 (Timelike)
static inline double compute_proper_velocity_invariant(GeodesicState4D s, double M) {
    double r = s.x[1];
    double f = 1.0 - (2.0 * M / r);
    double dt = s.u[0];
    double dr = s.u[1];
    double dphi = s.u[3];
    return - f * (dt * dt) + (1.0 / f) * (dr * dr) + (r * r) * (dphi * dphi);
}

int main() {
    printf("%s\n", "=== GEODESIC 4-VECTOR INTEGRATION (Circular Orbit at r = 6M) ===");
    GeodesicState4D s0 = {{0.0, 6.0, 1.570796, 0.0}, {1.41421356, 0.0, 0.0, 0.09622504}};
    printf("%s\n", "Initial Radial Coordinate (r0):");
    double r0 = s0.x[1];
    printf("[Manifold Invariant]: %.10f\n", (double)(r0));
    printf("%s\n", "Timelike 4-Velocity Invariant (g_ab u^a u^b = -1.0):");
    double norm0 = compute_proper_velocity_invariant(s0, 1.0);
    printf("[Manifold Invariant]: %.10f\n", (double)(norm0));
    GeodesicState4D s1 = geodesic_step(s0, 1.0, 0.1);
    printf("%s\n", "Radial Coordinate after Geodesic Step (r1):");
    double r1 = s1.x[1];
    printf("[Manifold Invariant]: %.10f\n", (double)(r1));
    printf("%s\n", "Conserved Timelike Invariant after step:");
    double norm1 = compute_proper_velocity_invariant(s1, 1.0);
    printf("[Manifold Invariant]: %.10f\n", (double)(norm1));
    return 0;
}