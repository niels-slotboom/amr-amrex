#pragma once
#include "AMReX.H"
#include "int/Integrator.hpp"

struct HeatEqnRHS {
    static constexpr int ncomp = 1;
    static constexpr int ngrow = 1;

    static inline const amrex::Vector<std::string> comp_names = {"phi"};

    enum Component : int { phi = 0 };

    AMREX_GPU_HOST_DEVICE
    explicit HeatEqnRHS(amrex::Real dx) : inv_dx_sq(1.0 / (dx * dx)) {}

    amrex::Real inv_dx_sq;

    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                           amrex::Array4<const amrex::Real> const& arr) const {
        switch (comp) {
        case Component::phi: {
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

static_assert(RHSConcept<HeatEqnRHS>);