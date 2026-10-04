#pragma once
#include "AMReX_GpuDevice.H"
#include "Integrator.hpp"

#pragma once
#include "AMReX_Array4.H"
#include "AMReX_MFIter.H"
#include "int/Integrator.hpp"

template <typename RHSFunctor, typename InitFunctor>
    requires RHSConcept<RHSFunctor> && InitConcept<InitFunctor>
class RK4Integrator : public Integrator<RHSFunctor, InitFunctor> {
    using Base = Integrator<RHSFunctor, InitFunctor>;
    // bring member constants into class scope
    using Base::ncomp;
    using Base::ngrow;

    // bring protected members into scope
    using Base::ba;
    using Base::dm;
    using Base::geom;
    using Base::rhs;
    using Base::state_new;
    using Base::state_old;
    using Base::time;

  public:
    RK4Integrator() = delete;

    // Forward constructor
    RK4Integrator(amrex::Geometry geom_, const amrex::IntVect& block_size, RHSFunctor rhs_, InitFunctor init_)
        : Base(std::move(geom_), block_size, std::move(rhs_), std::move(init_)), init(Base::state_old),
          acc(Base::state_new), arg(ba, dm, ncomp, ngrow), stage(ba, dm, ncomp, ngrow) {}

    ~RK4Integrator() = default;

  protected:
    void computeNewState(amrex::Real delta_time) override {
        amrex::Real dt = delta_time;
        amrex::Real dt_2 = (1.0 / 2.0) * dt;
        amrex::Real dt_3 = (1.0 / 3.0) * dt;
        amrex::Real dt_6 = (1.0 / 6.0) * dt;

        // k1 eval + accumulation
        computeStage(time, 0.0, true); // stage <- F(init)
                                       //     ==> stage = k1
        accumulateStage(dt_6, true);   // acc   <- init + s/6 * stage
                                       //     ==> acc = init + s/6 * k1

        // k2 eval + accumulation
        computeStage(time + dt_2, dt_2, false); // stage <- F(init + s/2 * stage)
                                                //     ==> stage = k2
        accumulateStage(dt_3, false);           // acc   <- acc + s/3 * stage
                                                //     ==> acc = init + s/6 * (k1 + 2 * k2)

        // k3 eval + accumulation
        computeStage(time + dt_2, dt_2, false); // stage <- F(init + s/2 * stage)
                                                //     ==> stage = k3
        accumulateStage(dt_3, false);           // acc   <- acc + s/3 * stage
                                                //     ==> acc = init + s/6 * (k1 + 2 * k2 + 2 * k3)

        // k4 eval + accumulation
        computeStage(time + dt, dt, false); // stage <- F(init + s * stage)
                                            //     ==> stage = k4
        accumulateStage(dt_6, false);       // acc   <- acc + s/6 * stage
                                            //     ==> acc = init + s/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    }

  private:
    void computeStage(amrex::Real eval_time, amrex::Real coeff, bool is_first) {
        // --- prepare arg for stage evaluation ---
        const amrex::MultiFab* arg_for_eval_ptr = &init; // if is_first, otherwise will get overwritten'

        if (!is_first) { // if not the first stage, so RHS arg needs to be computed into arg
                         // arg = init + coeff * stage (previous)
            amrex::MultiFab::LinComb(arg, 1.0, init, 0, coeff, stage, 0, 0, ncomp, 0);
            arg.FillBoundary(geom.periodicity());
            arg_for_eval_ptr = &arg;
        }
        const amrex::MultiFab& arg_for_eval = *arg_for_eval_ptr; // lock into a reference for cleaner code from now on

        // --- evaluate stage ---
        RHSFunctor rhs_ = rhs; // local copy for lambda capture
        for (amrex::MFIter mfi(stage); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            amrex::Array4<amrex::Real> arr_stage = stage.array(mfi);
            amrex::Array4<const amrex::Real> arr_arg = arg_for_eval.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int comp = 0; comp < ncomp; ++comp) {
                    arr_stage(i, j, k, comp) = rhs_(i, j, k, comp, eval_time, arr_arg);
                }
            });
        }
    }

    void accumulateStage(amrex::Real coeff, bool is_first) {
        if (is_first) // acc = init + coeff * stage
            amrex::MultiFab::LinComb(acc, 1.0, init, 0, coeff, stage, 0, 0, ncomp, 0);
        else // acc = acc + coeff * stage
            amrex::MultiFab::LinComb(acc, 1.0, acc, 0, coeff, stage, 0, 0, ncomp, 0);
    }

  private: // total of four registers required for RK4:
    // base class MultiFab name aliases
    const amrex::MultiFab& init; ///< Reference to Integrator::state_old
    amrex::MultiFab& acc;        ///< Reference to Integrator::state_new

    // implementation-specific temporary registers
    amrex::MultiFab arg;   ///< Temporary storage for RHS evaluation arguments
    amrex::MultiFab stage; ///< Last stage evaluation
};