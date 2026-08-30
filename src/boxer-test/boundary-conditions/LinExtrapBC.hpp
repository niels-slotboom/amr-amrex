#pragma once

#include "AMReX_BC_TYPES.H"
#include "AMReX_IntVect.H"
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

struct LinExtrapBC {
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
        amrex::IntVect boundary = nearest_interior(iv, geom.Domain());
        amrex::IntVect interior = reflect(iv, boundary);

        for (int n = 0; n < numcomp; ++n) {
            int comp = dcomp + n;
            dest(AMREX_D_DECL(iv[0], iv[1], iv[2]), comp) =
                2.0 * dest(AMREX_D_DECL(boundary[0], boundary[1], boundary[2]), comp) -
                dest(AMREX_D_DECL(interior[0], interior[1], interior[2]), comp);
        }
    }

  private:
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect nearest_interior(const amrex::IntVect& iv,
                                                                        const amrex::Box& dom) const {
        amrex::IntVect result = iv;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[d] = amrex::Clamp(iv[d], dom.smallEnd(d), dom.bigEnd(d));
        }
        return result;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::IntVect reflect(const amrex::IntVect& iv,
                                                               const amrex::IntVect& reflector) const {
        return 2 * reflector - iv;
    }
};