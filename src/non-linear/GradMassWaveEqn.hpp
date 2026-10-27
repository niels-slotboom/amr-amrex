#pragma once

#include "AMReX_RealVect.H"
#include "init/FunctionInit.hpp"
#include "pde-model/PDEModel.hpp"
#include "pde-model/PDEModelConcepts.hpp"
#include "pde-model/rhs-decorators/KODecorator.hpp"
#include "util/stencil.hpp"

template <int ngrow_> struct GradMassRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

    amrex::Real gamma = 0.0;

    amrex::Real inv_dx_sq = 0.0;

    enum Components : int { phi = 0, pi = 1 };

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        using namespace treesap::stencil;
        switch (comp) {
        case Components::phi:
            return arr(i, j, k, Components::pi);
        case Components::pi:
            return inv_dx_sq *
                   (laplacian<ngrow>(i, j, k, Components::phi, arr) +
                    gamma * gradient_squared<ngrow>(i, j, k, Components::phi, arr) * arr(i, j, k, Components::phi));
        default:
            return NAN;
        }
    }
};

static_assert(treesap::RHSConcept<GradMassRHS<1>>);

template <int ngrow> auto inline get_model(amrex::Real gamma) {
    using RHSType = GradMassRHS<ngrow>;
    using DecoratedRHSType = treesap::KODecorator<RHSType>;
    constexpr int ncomp = RHSType::ncomp;
    using InitType = treesap::FunctionInit<ncomp>;
    using ModelType = treesap::PDEModel<DecoratedRHSType, InitType>;

    amrex::Box box({0, 0, 0}, {512, 512, 512});
    amrex::RealBox rbox({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5});
    int is_per[] = {1, 1, 1};
    amrex::Geometry geom(box, &rbox, amrex::CoordSys::cartesian, is_per);
    amrex::Real dx = geom.CellSize(0);

    RHSType rhs{gamma, 1.0 / (dx * dx)};
    InitType init({"exp(-(x^2 + y^2 + z^2) / 0.1^2)", "0.0"});
    // InitType init({"exp(-x^2 / 0.1^2)", "0.0"});

    DecoratedRHSType rhs_dec{rhs, 1.0, 1.0 / dx};

    amrex::AllPrint() << "gamma = " << rhs.gamma << std::endl;
    amrex::AllPrint() << "dx_inv = " << rhs_dec.dx_inv << std::endl;
    amrex::AllPrint() << "strength = " << rhs_dec.strength << std::endl;

    ModelType model({"phi", "pi"}, std::move(geom), std::move(rhs_dec), std::move(init));

    return model;
}
