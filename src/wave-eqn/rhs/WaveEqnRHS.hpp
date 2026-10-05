#pragma once
#include "AMReX.H"
#include "int/Integrator.hpp"
#include "util/stencil.hpp"

template <int ngrow_> struct WaveEqnRHS {
    static_assert(ngrow_ > 0, "ngrow must be positive");
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

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
            using namespace stencil;
            return inv_dx_sq * laplacian<ngrow>(i, j, k, Component::phi, arr);
        }

        default:
            return NAN;
        }
    }
};

static_assert(RHSConcept<WaveEqnRHS<1>>);