#pragma once
#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>

#include <concepts>

#include "bcs/NoOpBCFunctor.hpp"
#include "util/PlotExporter.hpp"

/**
 * @brief Concept verifying that a type can evaluate a Right-Hand Side (RHS) term.
 * @details Requires the type to implement operator() with the correct signature and statically
 *          expose `ncomp` and `ngrow` as integer constants.
 */
template <typename T>
concept RHSConcept =
    std::is_trivially_copyable_v<T> &&
    requires(const T& t, int i, int j, int k, int comp, amrex::Real time, amrex::Array4<const amrex::Real> arr) {
        { t(i, j, k, comp, time, arr) } -> std::convertible_to<amrex::Real>;
        { T::comp_names } -> std::convertible_to<amrex::Vector<std::string>>;
        { T::ncomp } -> std::convertible_to<int>;
        { T::ngrow } -> std::convertible_to<int>;
    };

/**
 * @brief Concept verifying that a type can evaluate initial conditions at given spatial coordinates.
 * @details Requires the type to implement operator() with the correct signature and statically
 *          expose `ncomp` as an integer constant.
 */
template <typename T>
concept InitConcept =
    std::copy_constructible<T> && requires(const T& t, amrex::Real x, amrex::Real y, amrex::Real z, int comp) {
        { t(x, y, z, comp) } -> std::convertible_to<amrex::Real>;
        { T::ncomp } -> std::convertible_to<int>;
    };

template <typename T>
concept BCConcept = std::is_trivially_copyable_v<T> &&
                    requires(const T& t, const amrex::IntVect& iv,   // 1. Grid cell index (i, j, k) being filled
                             const amrex::Array4<amrex::Real>& dest, // 2. View into the destination FAB data buffer
                             int dcomp,                              // 3. Starting component index in 'dest' to fill
                             int numcomp,                            // 4. Number of components to process in this pass
                             const amrex::GeometryData& geom, // 5. GPU-safe domain geometry (box boundaries, dx, etc.)
                             amrex::Real time,        // 6. Current simulation time (useful for time-dependent BCs)
                             const amrex::BCRec* bcr, // 7. Array of boundary condition specifiers per component
                             int bcomp,               // 8. Component index offset into 'bcr'
                             int orig_comp            // 9. Original component index in the source MultiFab
                    ) {
                        { t(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp) } -> std::same_as<void>;
                        { T::ncomp } -> std::convertible_to<int>;
                        { T::ngrow } -> std::convertible_to<int>;
                    };

/**
 * @brief Base class for time-integration frameworks handling grid layout, state containers, and execution lifecycle.
 * @details Automatically infers component count (`ncomp`) and ghost cell requirements (`ngrow`)
 *          statically from the provided RHSFunctor. Functor structs must declare these as static members.
 *
 *          Derived classes implementing specific integration schemes (e.g., multi-stage Runge-Kutta)
 *          are responsible for allocating and managing any auxiliary state memory or intermediate
 *          buffers required by their algorithm, as well as orchestrating necessary intermediate
 *          ghost-cell exchanges (`FillBoundary`).
 *
 * @tparam RHSFunctor Type representing the spatial RHS operator (must satisfy RHSConcept).
 * @tparam InitFunctor Type representing the initial condition provider (must satisfy InitConcept).
 */
template <typename RHSFunctor, typename InitFunctor,
          typename BCFunctor = NoOpBCFunctor<RHSFunctor::ncomp, RHSFunctor::ngrow>>
    requires RHSConcept<RHSFunctor> && InitConcept<InitFunctor> && BCConcept<BCFunctor>
class Integrator {
  protected:
    static constexpr int ncomp = RHSFunctor::ncomp;
    static constexpr int ngrow = RHSFunctor::ngrow;

    static_assert(InitFunctor::ncomp == ncomp, "InitFunctor::ncomp does not match RHSFunctor::ncomp");
    static_assert(BCFunctor::ncomp == ncomp, "BCFunctor::ncomp does not match RHSFunctor::ncomp");
    static_assert(BCFunctor::ngrow == ngrow, "BCFunctor::ngrow does not match RHSFunctor::ngrow");

  public:
    /**
     * @brief Construct a new Integrator object.
     * @param geom_ AMReX Geometry specifying the physical domain.
     * @param block_size Maximum block size for box partitioning.
     * @param rhs_ Instance of the right-hand side (RHS) functor.
     * @param init_ Instance of the initialization functor.
     * @param bc_funct_ Instance of the boundary condition (BC) functor.
     * @param bc_recs_ Boundary condition records specifying which boundaries have which conditions
     */
    Integrator(amrex::Geometry geom_, const amrex::IntVect& block_size, RHSFunctor rhs_, InitFunctor init_,
               BCFunctor bc_funct_ = BCFunctor{}, amrex::Vector<amrex::BCRec> bc_recs_ = defaultBCRecs())
        : geom(std::move(geom_)), ba(makeBoxArray(geom, block_size)), dm(ba), state_old(ba, dm, ncomp, ngrow),
          state_new(ba, dm, ncomp, ngrow), rhs(std::move(rhs_)), init(std::move(init_)), bc_funct(bc_funct_),
          bc_recs(bc_recs_) {}

    virtual ~Integrator() = default;

    /**
     * @brief Initialise the primary state container using the initialization functor over physical coordinates.
     */
    void initialise() {
        InitFunctor init_ = init;
        amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();
        amrex::GpuArray<amrex::Real, 3> lo = geom.ProbLoArray();

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
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

    void configureExporter(const fs::path& path, int export_interval) {
        exporter.emplace(state_old, geom, RHSFunctor::comp_names, path, export_interval);
    }

    /**
     * @brief Execute a single time step of duration delta_time.
     * @param delta_time Size of the time step.
     */
    void step(amrex::Real delta_time) {
        FillPatch(state_old, time);

        if (exporter)
            exporter->exportIfNecessary(steps, time);

        computeNewState(delta_time);
        std::swap(state_old, state_new);

        steps++;
        time += delta_time;
    }

  protected:
    // Method to provide derived classes with a unified way to fill interior and exterior boundaries
    // with the BC functor managed by this base class.
    void FillPatch(amrex::MultiFab& mf, amrex::Real time_) {
        amrex::Vector<amrex::MultiFab*> src{&mf};
        amrex::Vector<amrex::Real> times{time_};

        // Instantiate GPU wrapper around the user's BCFunctor
        amrex::GpuBndryFuncFab<BCFunctor> gpu_bndry_func(bc_funct);
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCFunctor>> phys_bc(geom, bc_recs, gpu_bndry_func);

        amrex::FillPatchSingleLevel(mf, time_, src, times, 0, 0, ncomp, geom, phys_bc, 0);
    }

    /**
     * @brief Populate state_new based on state_old over a given time step.
     * @param delta_time Size of the time step.
     */
    virtual void computeNewState(amrex::Real delta_time) = 0;

    // Default helper to build a default BCRec vector (e.g. periodic/INT_DIR)
    static amrex::Vector<amrex::BCRec> defaultBCRecs() {
        amrex::Vector<amrex::BCRec> recs(ncomp);
        for (int n = 0; n < ncomp; ++n) {
            for (int dir = 0; dir < 3; ++dir) {
                recs[n].setLo(dir, amrex::BCType::int_dir); // default = interior/periodic
                recs[n].setHi(dir, amrex::BCType::int_dir); // default = interior/periodic
            }
        }
        return recs;
    }

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
    BCFunctor bc_funct;
    amrex::Vector<amrex::BCRec> bc_recs;

    std::optional<PlotExporter> exporter = std::nullopt; ///< Exporter for periodic plotfile export
};