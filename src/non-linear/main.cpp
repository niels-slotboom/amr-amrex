#include "AMReX.H"
#include "AMReX_Print.H"

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::AllPrint() << "Hello, World!" << std::endl;
    }
    amrex::Finalize();
}