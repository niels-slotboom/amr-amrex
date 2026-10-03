#pragma once
#include "AMReX.H"
#include "int/Integrator.hpp"

struct WaveEqnRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = 1;

    static inline const amrex::Vector<std::string> comp_names = {"phi", "dphi_dt"};

    enum Component : int { phi = 0, dphi_dt = 1 };

    AMREX_GPU_HOST_DEVICE
    explicit WaveEqnRHS(amrex::Real dx) : inv_dx_sq(1.0 / (dx * dx)) {}

    amrex::Real inv_dx_sq;

    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                           amrex::Array4<const amrex::Real> const& arr) const {
        switch (comp) {
        case Component::phi:
            return arr(i, j, k, Component::dphi_dt);

        case Component::dphi_dt: {
            amrex::Real stencil = -6.0 * arr(i, j, k, Component::phi);
            stencil += arr(i + 1, j, k, Component::phi) + arr(i - 1, j, k, Component::phi);
            stencil += arr(i, j + 1, k, Component::phi) + arr(i, j - 1, k, Component::phi);
            stencil += arr(i, j, k + 1, Component::phi) + arr(i, j, k - 1, Component::phi);
            return inv_dx_sq * stencil;
        }

        default:
            return NAN;
        }
    }
};

static_assert(RHSConcept<WaveEqnRHS>);