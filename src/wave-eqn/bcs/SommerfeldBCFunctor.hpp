#pragma once

struct SommerfeldBCFunctor {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = 1;

    AMREX_GPU_HOST_DEVICE
    void operator()(amrex::IntVect const& iv, amrex::Array4<amrex::Real> const& dest, int dcomp, int numcomp,
                    amrex::GeometryData const& geom, amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const {
        // TODO: Implement
    }
};
