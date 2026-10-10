#pragma once

template <int ncomp_, int ngrow_> struct NoOpBCFunctor {
    static constexpr int ncomp = ncomp_;
    static constexpr int ngrow = ngrow_;

    AMREX_GPU_DEVICE
    void operator()(amrex::IntVect const& iv, amrex::Array4<amrex::Real> const& dest, int dcomp, int numcomp,
                    amrex::GeometryData const& geom, amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const {
        // do nothing (its NoOp duh)
    }
};
