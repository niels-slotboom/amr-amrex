#pragma once
#include "AMReX_MFIter.H"
#include "Integrator.hpp"

template <typename RHSFunctor, typename InitFunctor,
          typename BCFunctor = NoOpBCFunctor<RHSFunctor::ncomp, RHSFunctor::ngrow>>
    requires RHSConcept<RHSFunctor> && InitConcept<InitFunctor> && BCConcept<BCFunctor>
class RK4Integrator : public Integrator<RHSFunctor, InitFunctor, BCFunctor> {
    using Base = Integrator<RHSFunctor, InitFunctor, BCFunctor>;

  private: // import base class members to scope
    using Base::ba;
    using Base::defaultBCRecs;
    using Base::dm;
    using Base::FillPatch;
    using Base::geom;
    using Base::ncomp;
    using Base::ngrow;
    using Base::rhs;
    using Base::state_new;
    using Base::state_old;
    using Base::steps;
    using Base::time;

  private: // coefficients
    static constexpr int stage_count = 4;
    static constexpr amrex::Array<amrex::Real, stage_count> a_fraction_arr{0.0, 0.5, 0.5, 1.0}; // arg offset coeffs
    static constexpr amrex::Array<amrex::Real, stage_count> b_fraction_arr{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                                                           1.0 / 6.0}; // accumulation coeffs

  public: // public interface
    RK4Integrator() = delete;
    RK4Integrator(amrex::Geometry geom_, const amrex::IntVect& block_size, RHSFunctor rhs_, InitFunctor init_,
                  BCFunctor bc_funct_ = BCFunctor{}, amrex::Vector<amrex::BCRec> bc_recs_ = defaultBCRecs())
        : Base(std::move(geom_), block_size, std::move(rhs_), std::move(init_), std::move(bc_funct_),
               std::move(bc_recs_)),
          init(state_old), acc(state_new), arg(ba, dm, ncomp, ngrow), arg_next(ba, dm, ncomp, ngrow) {}

    void computeNewState(amrex::Real delta_time) override {
        runStage<0>(delta_time);
        runStage<1>(delta_time);
        runStage<2>(delta_time);
        runStage<3>(delta_time);
    }

  private: // fused stage kernel
    template <int stage_number> void runStage(amrex::Real dt) {
        constexpr bool is_first = (stage_number == 0);
        constexpr bool is_last = (stage_number == (stage_count - 1));

        amrex::Real a_next = dt * a_fraction_arr[(stage_number + 1) % stage_count];
        amrex::Real time_arg = time + dt * a_fraction_arr[stage_number];
        amrex::Real b = dt * b_fraction_arr[stage_number];

        if constexpr (!is_first) { // finalise arg for rhs evaluation
            std::swap(arg_next, arg);
            FillPatch(arg, time_arg);
        }

        RHSFunctor rhs_ = rhs; // local copy for lambda capture

        for (amrex::MFIter mfi(init); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            amrex::Array4<const amrex::Real> init_arr = init.const_array(mfi);
            amrex::Array4<amrex::Real> acc_arr = acc.array(mfi);
            amrex::Array4<const amrex::Real> arg_arr = arg.const_array(mfi);
            amrex::Array4<amrex::Real> arg_next_arr = arg_next.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int comp = 0; comp < ncomp; ++comp) {
                    // compute stage k_i for current voxel and accumulate
                    amrex::Real stage;
                    if constexpr (is_first) {
                        stage = rhs_(i, j, k, comp, time_arg, init_arr);
                        acc_arr(i, j, k, comp) = init_arr(i, j, k, comp) + b * stage;
                    } else {
                        stage = rhs_(i, j, k, comp, time_arg, arg_arr);
                        acc_arr(i, j, k, comp) += b * stage;
                    }

                    // prepare arg_next if there is a next stage
                    if constexpr (!is_last) {
                        arg_next_arr(i, j, k, comp) = init_arr(i, j, k, comp) + a_next * stage;
                    }
                }
            });
        }
    }

  private: // class members
    // aliases for existing base class registers
    const amrex::MultiFab& init; ///< Alias for state_old, holds previous state
    amrex::MultiFab& acc;        ///< Alias for state_new, holds (previous state + stages)
    // temporary registers for more efficient evaluation
    amrex::MultiFab arg;      ///< Linear combination of arguments for the current RHS eval
    amrex::MultiFab arg_next; ///< Linear combination of arguments for the next RHS eval
};