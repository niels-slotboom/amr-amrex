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

        AMRContainer amr(
            geom, amr_info,
            "1/sqrt(x*x + y*y + z*z + 0.0001) + 1/sqrt((x-0.4)*(x-0.4) + (y-0.3)*(y-0.3) + 0.1 * z*z + 0.0001)",
            {"x", "y", "z"}, {"phi"}, ngrow);
        amr.InitFromScratch(0.0);

        amrex::Vector<int> level_steps(amr_info.max_level, 0);

        amr.writeMultiLevelPlotFile("raw/test", 0.0, level_steps);

        boxer::show(amr, ngrow);
    }
    // Clean up resources
    amrex::Finalize();
    return 0;
}