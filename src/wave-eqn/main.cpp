#include "AMReX.H"
#include "AMReX_CoordSys.H"
#include "AMReX_IntVect.H"
#include "AMReX_Print.H"
#include "init/ConstInit.hpp"
#include "init/FunctionInit.hpp"
#include "int/EulerIntegrator.hpp"
#include "rhs/WaveEqnRHS.hpp"
#include <iostream>

template <typename T> std::string duration_since(T start) {
    // 1. Calculate duration in milliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);

    // 2. Extract hours, minutes, seconds, and milliseconds
    auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed);
    elapsed -= hours;

    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed);
    elapsed -= minutes;

    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
    elapsed -= seconds;

    auto millis = elapsed; // remaining milliseconds

    // 3. Format into [hh:mm:ss.xxx]
    std::ostringstream ss;
    ss << '[' << std::setfill('0') << std::setw(2) << hours.count() << ':' << std::setfill('0') << std::setw(2)
       << minutes.count() << ':' << std::setfill('0') << std::setw(2) << seconds.count() << '.' << std::setfill('0')
       << std::setw(3) << millis.count() << ']';

    return ss.str();
}

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        using RHS = WaveEqnRHS;
        using Init = FunctionInit<RHS::ncomp>;

        amrex::AllPrint() << "Setting up simulation..." << std::endl;

        // set up domain
        amrex::Box box({0, 0, 0}, {256, 256, 256});
        amrex::RealBox rbox({-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0});
        int is_per[] = {1, 1, 1};
        amrex::Geometry geom(box, &rbox, amrex::CoordSys::cartesian, is_per);

        amrex::Real dx = geom.CellSize(0);

        RHS rhs(dx);
        Init init({"0.0", "0.0"});

        EulerIntegrator<RHS, Init> integrator(geom, {96, 96, 96}, rhs, init);

        integrator.initialise();

        amrex::AllPrint() << "Initialisation done." << std::endl;

        size_t steps = 1e4;
        size_t export_interval = 100;

        auto start = std::chrono::system_clock::now();

        for (size_t step = 0; step <= steps; ++step) {
            if (step % export_interval == 0) {
                amrex::AllPrint() << duration_since(start) << " Reached step " << step << std::endl;
            }

            integrator.step(std::sqrt(0.5) * dx);
        }
    }
    amrex::Finalize();
}