#pragma once
#include "AMReX.H"
#include "int/Integrator.hpp"
#include "util/stencil.hpp"

template <int ngrow_> struct HeatEqnRHS {
    static constexpr int ncomp = 1;
    static constexpr int ngrow = ngrow_;

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
            using namespace stencil;
            return inv_dx_sq * laplacian<ngrow_>(i, j, k, 0, arr);
        }

        default:
            return NAN;
        }
    }
};

static_assert(RHSConcept<HeatEqnRHS<1>>);