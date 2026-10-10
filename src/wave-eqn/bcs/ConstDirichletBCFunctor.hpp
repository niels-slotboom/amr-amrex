#pragma once
#include "int/Integrator.hpp"

template <int ncomp_, int ngrow_> struct ConstDirichletBCFunctor {
    static constexpr int ncomp = ncomp_;
    static constexpr int ngrow = ngrow_;

    amrex::GpuArray<amrex::Real, ncomp> boundary_values{};

    AMREX_GPU_DEVICE
    void operator()(amrex::IntVect const& iv, amrex::Array4<amrex::Real> const& dest, int dcomp, int numcomp,
                    amrex::GeometryData const& geom, amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const {
        const amrex::Box& domain = geom.Domain();

        // check if iv is outside domain
        bool is_outside = false;
        for (int dir = 0; dir < 3; ++dir) {
            if (iv[dir] < domain.smallEnd(dir) || iv[dir] > domain.bigEnd(dir)) {
                is_outside = true;
                break;
            }
        }

        if (!is_outside)
            return;

        for (int comp = 0; comp < ncomp; ++comp) {
            dest(iv, comp) = boundary_values[comp];
        }
    }
};
