#pragma once
#include "AMReX.H"
#include "AMReX_RealVect.H"
#include "int/Integrator.hpp"

struct NLGradDampWaveEqnRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = 1;

    static inline const amrex::Vector<std::string> comp_names = {"phi", "dphi_dt"};

    enum Component : int { phi = 0, dphi_dt = 1 };

    AMREX_GPU_HOST_DEVICE
    explicit NLGradDampWaveEqnRHS(amrex::Real dx, amrex::Real gamma_)
        : inv_dx_sq(1.0 / (dx * dx)), inv_dx_sq_4(0.25 * inv_dx_sq), gamma(gamma_) {}

    const amrex::Real inv_dx_sq;
    const amrex::Real inv_dx_sq_4;
    const amrex::Real gamma;

    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                           amrex::Array4<const amrex::Real> const& arr) const {
        switch (comp) {
        case Component::phi:
            return arr(i, j, k, Component::dphi_dt);

        case Component::dphi_dt: {
            amrex::Real lap_stencil = -6.0 * arr(i, j, k, Component::phi);
            lap_stencil += arr(i + 1, j, k, Component::phi) + arr(i - 1, j, k, Component::phi);
            lap_stencil += arr(i, j + 1, k, Component::phi) + arr(i, j - 1, k, Component::phi);
            lap_stencil += arr(i, j, k + 1, Component::phi) + arr(i, j, k - 1, Component::phi);

            amrex::Real dphi_dx_stencil = arr(i + 1, j, k, Component::phi) - arr(i - 1, j, k, Component::phi);
            amrex::Real dphi_dy_stencil = arr(i, j + 1, k, Component::phi) - arr(i, j - 1, k, Component::phi);
            amrex::Real dphi_dz_stencil = arr(i, j, k + 1, Component::phi) - arr(i, j, k - 1, Component::phi);
            amrex::Real grad_sq = inv_dx_sq_4 * (dphi_dx_stencil * dphi_dx_stencil + dphi_dy_stencil * dphi_dy_stencil +
                                                 dphi_dz_stencil * dphi_dz_stencil);

            return inv_dx_sq * lap_stencil - gamma * grad_sq * arr(i, j, k, Component::dphi_dt);
        }

        default:
            return NAN;
        }
    }
};

static_assert(RHSConcept<NLGradDampWaveEqnRHS>);