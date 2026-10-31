#include "AMReX.H"
#include "AMReX_Print.H"
#include "GradMassWaveEqn.hpp"
#include "integrators/RK4Integrator.hpp"
#include "util/ScopedTimer.hpp"

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        treesap::ScopedTimer::enabled = true;
        auto model = get_model<2>(10.0);
        treesap::RK4Integrator<decltype(model)> integrator(model, amrex::IntVect{128, 128, 128});

        size_t max_steps = 100000;
        size_t export_interval = 100;

        integrator.configureExporter("raw/nl_test", export_interval);

        amrex::Real dx = model.get_geometry().CellSize(0);
        amrex::Real dt = 0.0035 * dx;

        integrator.initialise();
        for (size_t step = 0; step <= max_steps; ++step) {
            amrex::AllPrint() << "Step " << step << std::endl;
            integrator.step(dt);
        }
    }
    amrex::Finalize();
}