#pragma once
#include "AMReX.H"
#include "int/Integrator.hpp"

template <int ncomp_> struct ConstInit {
    static_assert(ncomp_ > 0, "ConstInit must initialise a positive number of components.");
    static constexpr int ncomp = ncomp_;
    amrex::Real value;

    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z, int comp) const { return value; }
};

static_assert(InitConcept<ConstInit<1>>);
