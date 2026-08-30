#pragma once

#include "AMReX_BC_TYPES.H"
#include "AMReX_IntVect.H"
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

struct DirichletBC {
    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect& iv,               // 1. Grid cell index (i, j, k) being filled
                    const amrex::Array4<amrex::Real>& dest, // 2. View into the destination FAB data buffer
                    int dcomp,                              // 3. Starting component index in 'dest' to fill
                    int numcomp,                            // 4. Number of components to process in this pass
                    const amrex::GeometryData& geom,        // 5. GPU-safe domain geometry (box boundaries, dx, etc.)
                    amrex::Real time,        // 6. Current simulation time (useful for time-dependent BCs)
                    const amrex::BCRec* bcr, // 7. Array of boundary condition specifiers per component
                    int bcomp,               // 8. Component index offset into 'bcr'
                    int orig_comp            // 9. Original component index in the source MultiFab
    ) const {
        int i = iv[0];
        int j = iv[1];
        int k = iv[2];

        if (is_outside(iv, geom.Domain())) {
            // Loop over the requested components and set the ghost cell values
            for (int n = 0; n < numcomp; ++n) {
                dest(i, j, k, dcomp + n) = 0.0;
            }
        }
    }

  private:
    inline AMREX_GPU_DEVICE bool is_outside(const amrex::IntVect& iv, const amrex::Box& dom) const {
        int i = iv[0];
        int j = iv[1];
        int k = iv[2];

        bool is_lo_x = (i < dom.smallEnd(0));
        bool is_hi_x = (i > dom.bigEnd(0));
        bool is_lo_y = (j < dom.smallEnd(1));
        bool is_hi_y = (j > dom.bigEnd(1));
        bool is_lo_z = (k < dom.smallEnd(2));
        bool is_hi_z = (k > dom.bigEnd(2));

        return is_lo_x || is_hi_x || is_lo_y || is_hi_y || is_lo_z || is_hi_z;
    }
};