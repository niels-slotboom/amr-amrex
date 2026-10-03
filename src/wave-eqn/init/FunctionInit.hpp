#pragma once
#include "int/Integrator.hpp"

#include <AMReX_Array.H>
#include <AMReX_Parser.H>
#include <string>
#include <utility>

template <int ncomp_> struct FunctionInit {
    static_assert(ncomp_ > 0, "FunctionInit must initialise a positive number of components.");
    static constexpr int ncomp = ncomp_;

    FunctionInit() = delete;

    // constructor creates host parsers and extracts the compiled executors
    explicit FunctionInit(amrex::Array<std::string, ncomp> exprs_)
        : exprs(std::move(exprs_)), parsers(std::make_shared<amrex::Array<amrex::Parser, ncomp>>()) {
        for (int i = 0; i < ncomp; ++i) {
            amrex::Parser& parser = (*parsers)[i];
            parser.define(exprs[i]);
            parser.registerVariables({"x", "y", "z"});
            executors[i] = parser.compile<3>();
        }
    }

    // device-friendly call operator (evaluates the compiled parser executor)
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z, int comp) const {
        return executors[comp](x, y, z);
    }

  private:
    amrex::Array<std::string, ncomp> exprs;
    std::shared_ptr<amrex::Array<amrex::Parser, ncomp>> parsers; // shared_ptr because they need to be heap-allocated
    amrex::GpuArray<amrex::ParserExecutor<3>, ncomp> executors;
};

static_assert(InitConcept<FunctionInit<1>>);