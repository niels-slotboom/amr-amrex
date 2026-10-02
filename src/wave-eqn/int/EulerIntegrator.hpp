#pragma once
#include "AMReX_Array4.H"
#include "AMReX_MFIter.H"
#include "int/Integrator.hpp"

template <typename RHSFunctor, typename InitFunctor>
    requires RHSConcept<RHSFunctor> && InitConcept<InitFunctor>
class EulerIntegrator : public Integrator<RHSFunctor, InitFunctor> {
    // bring member constants into class scope
    using Integrator<RHSFunctor, InitFunctor>::ncomp;
    using Integrator<RHSFunctor, InitFunctor>::ngrow;

    // bring protected members into scope
    using Integrator<RHSFunctor, InitFunctor>::state_old;
    using Integrator<RHSFunctor, InitFunctor>::state_new;
    using Integrator<RHSFunctor, InitFunctor>::rhs;
    using Integrator<RHSFunctor, InitFunctor>::time;

  public:
    // Inherit constructor
    using Integrator<RHSFunctor, InitFunctor>::Integrator;

  protected:
    virtual void computeNewState(double delta_time) override {
        // make local for lambda capture
        RHSFunctor rhs_ = rhs;
        amrex::Real time_ = time;

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            amrex::Array4<const amrex::Real> arr_old = state_old.const_array(mfi);
            amrex::Array4<amrex::Real> arr_new = state_new.array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int comp = 0; comp < ncomp; ++comp) {
                    arr_new(i, j, k, comp) = arr_old(i, j, k, comp) + delta_time * rhs_(i, j, k, comp, time_, arr_old);
                }
            });
        }
    }
};