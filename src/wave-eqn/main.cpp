#include "AMReX.H"
#include "AMReX_CoordSys.H"
#include "AMReX_IntVect.H"
#include "AMReX_Print.H"
#include "init/ConstInit.hpp"
#include "init/FunctionInit.hpp"
#include "int/EulerIntegrator.hpp"
#include "int/FusedRK4Integrator.hpp"
#include "int/RK4Integrator.hpp"
#include "rhs/HeatEqnRHS.hpp"
#include "rhs/NLGradDampWaveEqnRHS.hpp"
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

void waveEqn() {
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
    Init init({"exp(-(x^2+y^2+z^2)/(0.4^2)) * cos(30.0*x)",
               "exp(-(x^2+y^2+z^2)/(0.4^2)) * ((2.0/0.4^2) * x * cos(30.0*x) + 30.0 * sin(30.0*x))"});
    RK4Integrator<RHS, Init> integrator(geom, {128, 128, 128}, std::move(rhs), std::move(init));

    integrator.initialise();

    amrex::AllPrint() << "Initialisation done." << std::endl;

    size_t steps = 800;
    size_t export_interval = 2;
    integrator.configureExporter("raw/test", export_interval);

    auto start = std::chrono::system_clock::now();

    for (size_t step = 0; step <= steps; ++step) {
        if (step % export_interval == 0) {
            amrex::AllPrint() << duration_since(start) << " Reached step " << step << std::endl;
        }

        integrator.step(std::sqrt(2.0 / 3.0) * dx);
    }
}

void nonLinWaveEqn() {
    using RHS = NLGradDampWaveEqnRHS;
    using Init = FunctionInit<RHS::ncomp>;

    amrex::AllPrint() << "Setting up simulation..." << std::endl;

    // set up domain
    amrex::Box box({0, 0, 0}, {256, 256, 256});
    amrex::RealBox rbox({-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0});
    int is_per[] = {1, 1, 1};
    amrex::Geometry geom(box, &rbox, amrex::CoordSys::cartesian, is_per);

    amrex::Real dx = geom.CellSize(0);

    RHS rhs(dx, 0.1);
    Init init({"exp(-(x^2+y^2+z^2)/(0.4^2)) * cos(30.0*x)",
               "exp(-(x^2+y^2+z^2)/(0.4^2)) * ((2.0/0.4^2) * x * cos(30.0*x) + 30.0 * sin(30.0*x))"});
    FusedRK4Integrator<RHS, Init> integrator(geom, {128, 128, 128}, std::move(rhs), std::move(init));

    integrator.initialise();

    amrex::AllPrint() << "Initialisation done." << std::endl;

    size_t steps = 400;
    size_t export_interval = 2;
    integrator.configureExporter("raw/test", export_interval);

    auto start = std::chrono::system_clock::now();

    for (size_t step = 0; step <= steps; ++step) {
        if (step % export_interval == 0) {
            amrex::AllPrint() << duration_since(start) << " Reached step " << step << std::endl;
        }

        integrator.step(std::sqrt(2.0 / 3.0) * dx);
    }
}

void heatEqn() {
    using RHS = HeatEqnRHS;
    using Init = FunctionInit<RHS::ncomp>;

    amrex::AllPrint() << "Setting up simulation..." << std::endl;

    // set up domain
    amrex::Box box({0, 0, 0}, {256, 256, 256});
    amrex::RealBox rbox({-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0});
    int is_per[] = {1, 1, 1};
    amrex::Geometry geom(box, &rbox, amrex::CoordSys::cartesian, is_per);

    amrex::Real dx = geom.CellSize(0);

    RHS rhs(dx);
    Init init({"exp(-((x-1)^2+y^2+z^2)/(0.1^2)) + exp(-((x+1)^2+y^2+z^2)/(0.1^2))"});

    EulerIntegrator<RHS, Init> integrator(geom, {96, 96, 96}, std::move(rhs), std::move(init));

    integrator.initialise();

    amrex::AllPrint() << "Initialisation done." << std::endl;

    size_t steps = 1e5;
    size_t export_interval = 1e2;
    integrator.configureExporter("raw/test", export_interval);

    auto start = std::chrono::system_clock::now();

    for (size_t step = 0; step <= steps; ++step) {
        if (step % export_interval == 0) {
            amrex::AllPrint() << duration_since(start) << " Reached step " << step << std::endl;
        }

        integrator.step(dx * dx / 6.0);
    }
}

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    nonLinWaveEqn();
    amrex::Finalize();
}