#include "AMRContainer.hpp"
#include <Boxer/Boxer.hpp>

int main(int argc, char* argv[]) { // Initialize AMReX (handles MPI setup, GPU device selection, etc.)
    amrex::Initialize(argc, argv);
    {
        constexpr int dim = AMREX_SPACEDIM;

        // Index space: 32^3 cells
        amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), amrex::IntVect(AMREX_D_DECL(63, 63, 63)));

        // Physical domain: [0,1]^3
        amrex::RealBox real_box({AMREX_D_DECL(-2.0, -2.0, -2.0)}, {AMREX_D_DECL(2.0, 2.0, 2.0)});

        std::array<int, dim> is_periodic{AMREX_D_DECL(1, 1, 1)};

        amrex::Geometry geom(domain, &real_box, 0, is_periodic.data());

        // -----------------------------------------------------------------------------
        // AMR configuration
        // -----------------------------------------------------------------------------

        amrex::AmrInfo amr_info;

        amr_info.max_level = 10;

        auto ref_ratio = amrex::IntVect(AMREX_D_DECL(2, 2, 2));
        amr_info.ref_ratio = {ref_ratio, ref_ratio, ref_ratio, ref_ratio, ref_ratio};

        auto n_error_buf = amrex::IntVect(AMREX_D_DECL(3, 3, 3));
        amr_info.n_error_buf = {n_error_buf, n_error_buf, n_error_buf, n_error_buf, n_error_buf};

        auto blocking_factor = amrex::IntVect(AMREX_D_DECL(16, 16, 16));
        amr_info.blocking_factor = {blocking_factor, blocking_factor, blocking_factor, blocking_factor,
                                    blocking_factor};

        auto max_grid_size = amrex::IntVect(AMREX_D_DECL(32, 32, 32));
        amr_info.max_grid_size = {max_grid_size, max_grid_size, max_grid_size, max_grid_size, max_grid_size};

        // Construct your AmrCore derivative
        int ngrow = 1;

        AMRContainer amr(geom, amr_info, "1/sqrt(x*x + y*y + z*z + 0.001)", {"x", "y", "z"}, {"phi"}, ngrow);
        amr.InitFromScratch(0.0);

        const auto& mf = amr.getState(0);
        amrex::Real min_val = mf.min(0);
        amrex::Real max_val = mf.max(0);

        amrex::Print() << "Level 0 Min: " << min_val << " | Max: " << max_val << "\n";

        boxer::show(amr, ngrow);
    }
    // Clean up resources
    amrex::Finalize();
    return 0;
}