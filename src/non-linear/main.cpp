#include "AMReX.H"
#include "AMReX_Print.H"
#include "GradMassWaveEqn.hpp"
#include "integrators/RK4Integrator.hpp"

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::AllPrint() << "Hello, World!" << std::endl;

        auto model = get_model();
        RK4Integrator<decltype(model)> integrator(model, amrex::IntVect{64, 64, 64});

        integrator.configureExporter("raw/nl_test/", 10);

        size_t max_steps = 1000;
        amrex::Real dx = model.get_geometry().CellSize(0);
        amrex::Real dt = sqrt(2.0 / 3.0) * dx;

        for (size_t step = 0; step <= max_steps; ++step) {
            integrator.step(dt);
        }
    }
    amrex::Finalize();
}