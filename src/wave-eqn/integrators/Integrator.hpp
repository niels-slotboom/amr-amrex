#include "AMReX_DistributionMapping.H"
#include "AMReX_Geometry.H"
#include "AMReX_MFIter.H"
#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <concepts>

/**
 * @brief Concept verifying that a type can evaluate a Right-Hand Side (RHS) term.
 * @details Requires the type to implement operator() with the correct signature and statically
 *          expose `ncomp` and `ngrow` as integer constants.
 */
template <typename T>
concept RHSConcept =
    requires(T t, int i, int j, int k, int comp, amrex::Real time, amrex::Array4<const amrex::Real> arr) {
        { t(i, j, k, comp, time, arr) } -> std::convertible_to<amrex::Real>;
        { T::ncomp } -> std::convertible_to<int>;
        { T::ngrow } -> std::convertible_to<int>;
    };

/**
 * @brief Concept verifying that a type can evaluate initial conditions at given spatial coordinates.
 * @details Requires the type to implement operator() with the correct signature and statically
 *          expose `ncomp` as an integer constant.
 */
template <typename T>
concept InitConcept = requires(T t, amrex::Real x, amrex::Real y, amrex::Real z, int comp) {
    { t(x, y, z, comp) } -> std::convertible_to<amrex::Real>;
    { T::ncomp } -> std::convertible_to<int>;
};

/**
 * @brief Base class for time-integration frameworks handling grid layout, state containers, and execution lifecycle.
 * @details Automatically infers component count (`ncomp`) and ghost cell requirements (`ngrow`)
 *          statically from the provided RHSFunctor. Functor structs must declare these as static members.
 * @tparam RHSFunctor Type representing the spatial RHS operator (must satisfy RHSConcept).
 * @tparam InitFunctor Type representing the initial condition provider (must satisfy InitConcept).
 */
template <typename RHSFunctor, typename InitFunctor>
    requires RHSConcept<RHSFunctor> && InitConcept<InitFunctor>
class Integrator {
  protected:
    static constexpr int ncomp = RHSFunctor::ncomp;
    static constexpr int ngrow = RHSFunctor::ngrow;

    static_assert(InitFunctor::ncomp == ncomp, "InitFunctor::ncomp does not match RHSFunctor::ncomp");

  public:
    /**
     * @brief Construct a new Integrator object.
     * @param geom_ AMReX Geometry specifying the physical domain.
     * @param block_size Maximum block size for box partitioning.
     * @param rhs_ Instance of the RHS functor.
     * @param init_ Instance of the initialization functor.
     */
    Integrator(amrex::Geometry geom_, const amrex::IntVect& block_size, RHSFunctor rhs_, InitFunctor init_)
        : geom(std::move(geom_)), ba(makeBoxArray(geom, block_size)), dm(ba), state_old(ba, dm, ncomp, ngrow),
          state_new(ba, dm, ncomp, ngrow), rhs(std::move(rhs_)), init(std::move(init_)) {}

    /**
     * @brief Initialise the primary state container using the initialization functor over physical coordinates.
     */
    void initialise() {
        InitFunctor init_ = init;
        amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();
        amrex::GpuArray<amrex::Real, 3> lo = geom.ProbLoArray();

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            auto box = mfi.validbox();
            amrex::Array4<amrex::Real> arr = state_old.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real x = lo[0] + (i + 0.5) * dx[0];
                amrex::Real y = lo[1] + (j + 0.5) * dx[1];
                amrex::Real z = lo[2] + (k + 0.5) * dx[2];

                for (int comp = 0; comp < ncomp; comp++) {
                    arr(i, j, k, comp) = init_(x, y, z, comp);
                }
            });
        }
    }

    /**
     * @brief Execute a single time step of duration delta_time.
     * @param delta_time Size of the time step.
     */
    void step(double delta_time) {
        state_old.FillBoundary();

        computeNewState(delta_time);
        std::swap(state_old, state_new);

        steps++;
        time += delta_time;
    }

  protected:
    /**
     * @brief Populate state_new based on state_old over a given time step.
     * @param delta_time Size of the time step.
     */
    virtual void computeNewState(double delta_time) = 0;

  private:
    /**
     * @brief Helper to generate a partitioned BoxArray from domain geometry and block size.
     */
    static amrex::BoxArray makeBoxArray(const amrex::Geometry& geom, const amrex::IntVect& block_size) {
        amrex::BoxArray result(geom.Domain());
        result.maxSize(block_size);
        return result;
    }

  protected:
    amrex::Geometry geom;          ///< Physical domain geometry description.
    amrex::BoxArray ba;            ///< Box array partitioning the domain.
    amrex::DistributionMapping dm; ///< Distribution mapping across parallel ranks.

    amrex::MultiFab state_old; ///< State container at the current time level.
    amrex::MultiFab state_new; ///< State container at the advanced time level.

    size_t steps = 0;       ///< Total number of completed time steps.
    amrex::Real time = 0.0; ///< Current physical simulation time.

    RHSFunctor rhs; ///< Spatial PDE right-hand side evaluation functor.

  private:
    InitFunctor init; ///< Initial condition evaluation functor.
};