#pragma once
#include "AMReX.H"
#include "AMReX_RealVect.H"
#include "int/Integrator.hpp"
#include "util/stencil.hpp"

// ϕ_tt = Δϕ - γ|∇ϕ|²ϕ_t
template <int ngrow_> struct NLGradDampWaveEqnRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

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
            using namespace stencil;
            return inv_dx_sq * laplacian<ngrow>(i, j, k, Component::phi, arr) -
                   gamma * gradient_squared<ngrow>(i, j, k, Component::phi, arr) * arr(i, j, k, Component::dphi_dt);
        }

        default:
            return NAN;
        }
    }
};

static_assert(RHSConcept<NLGradDampWaveEqnRHS<1>>);