#include "AMRContainer.hpp"

#include "AMReX_Box.H"
#include "AMReX_FillPatchUtil.H"
#include "AMReX_MFIter.H"
#include "AMReX_Parser.H"

#include <ostream>
#include <sstream>
#include <stdexcept>

AMRContainer::AMRContainer(const amrex::Geometry& lev0_geom, const amrex::AmrInfo& amr_info, std::string initDataExpr_,
                           std::vector<std::string> initDataVars_, int nvar, int ngrow)
    : AmrCore(lev0_geom, amr_info), nvar(nvar), ngrow(ngrow),
      state(amr_info.max_level + 1), // Preallocate storage for up to max_level + 1 levels
      bcs(nvar), initDataVars(std::move(initDataVars_)), initDataExpr(std::move(initDataExpr_)), initDataExprParser() {

    // Initialize boundary condition metadata (defaulting to internal / periodic boundaries)
    for (int i = 0; i < nvar; ++i) {
        bcs[i] = amrex::BCRec(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir,
                              amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir);
    }

    // Set up initial data expression parser
    initDataExprParser.define(initDataExpr);
    initDataExprParser.registerVariables({initDataVars[0], initDataVars[1], initDataVars[2]});
}

void AMRContainer::printContainerInfo(int displayLimit) {
    size_t totalCells = 0;
    const int threshold = 20;

    amrex::AllPrint() << "-----------------------------------------------------" << std::endl;
    amrex::AllPrint() << "AMRContainer with " << finestLevel() << " levels" << std::endl;
    amrex::AllPrint() << "Initial data expr: f(" << initDataVars[0] << ", " << initDataVars[1] << ", "
                      << initDataVars[2] << ")" << initDataExpr << std::endl;

    for (int lev = 0; lev <= finestLevel(); lev++) {
        const amrex::BoxArray& ba = boxArray(lev);
        size_t totalCellsThisLev = ba.numPts();

        amrex::AllPrint() << "Level " << lev << " (Boxes: " << ba.size() << ", Cells: " << totalCellsThisLev << ")\n";

        std::ostringstream oss;
        int n = ba.size();

        for (int i = 0; i < n; i++) {
            // Print all boxes if under threshold, otherwise truncate middle boxes
            bool shouldPrint = (n <= threshold) || (i < displayLimit || i >= (n - displayLimit));

            if (shouldPrint) {
                const amrex::Box& bx = ba[i];
                oss << "  Box " << i << ": " << bx << " (" << bx.numPts() << " cells)\n";
            } else if (i == displayLimit) {
                oss << "  ... (" << (n - 2 * displayLimit) << " boxes omitted) ...\n";
            }
        }

        amrex::AllPrint() << oss.str() << std::flush;
        totalCells += totalCellsThisLev;
    }
    amrex::AllPrint() << "Total number of cells: " << totalCells << std::endl;
}

void AMRContainer::FillPatch(int lev, amrex::Real time) { FillPatch(state[lev], lev, time); }

void AMRContainer::FillPatch(amrex::MultiFab& dst, int lev, amrex::Real time) {
    if (lev < 0 || lev > finestLevel()) {
        amrex::Abort("AMRContainer::FillPatch: invalid level");
    }

    amrex::PhysBCFunctNoOp phys_bc; // Dummy functor for periodic / interior boundary handling

    if (lev == 0) {
        amrex::Vector<amrex::MultiFab*> src{&dst};
        amrex::Vector<amrex::Real> times{time};

        // Level 0: simple ghost-cell fill from periodic boundaries / neighboring grids
        amrex::FillPatchSingleLevel(dst, time, src, times, 0, 0, nvar, Geom(lev), phys_bc, 0);
    } else {
        amrex::Vector<amrex::MultiFab*> fine_src{&state[lev]};
        amrex::Vector<amrex::MultiFab*> coarse_src{&state[lev - 1]};
        amrex::Vector<amrex::Real> times{time};

        // Fine levels: fill interior from fine level, boundaries/ghosts interpolated from coarse level
        amrex::FillPatchTwoLevels(dst, time, coarse_src, times, fine_src, times, 0, 0, nvar, Geom(lev - 1), Geom(lev),
                                  phys_bc, 0, phys_bc, 0, refRatio(lev - 1), &amrex::cell_cons_interp, bcs, 0);
    }
}

const amrex::MultiFab& AMRContainer::getState(int lev) const {
    if (lev < 0 || lev > finestLevel()) {
        throw std::runtime_error("Requested level " + std::to_string(lev) + " does not exist");
    }
    return state[lev];
}

void AMRContainer::ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow_arg) {
    tags.setVal(amrex::TagBox::CLEAR);

    // Populate ghost cells so finite-difference stencil reads valid neighbor data
    FillPatch(lev, time);

    const int nvar_loc = nvar;
    const amrex::Real threshold = 0.01;

    // Iterate over grid patches on the GPU/CPU device
    for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        const auto& tags_arr = tags.array(mfi);
        const auto& arr = state[lev].array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real max_grad_sq = 0.0;

            // Estimate gradient squared across all variables
            for (int comp = 0; comp < nvar_loc; comp++) {
                amrex::Real dx = arr(i + 1, j, k, comp) - arr(i - 1, j, k, comp);
                amrex::Real dy = arr(i, j + 1, k, comp) - arr(i, j - 1, k, comp);
                amrex::Real dz = arr(i, j, k + 1, comp) - arr(i, j, k - 1, comp);

                amrex::Real grad_sq = dx * dx + dy * dy + dz * dz;
                max_grad_sq = amrex::max(max_grad_sq, grad_sq);
            }

            // Mark cell for refinement if gradient exceeds error tolerance
            if (max_grad_sq > threshold) {
                tags_arr(i, j, k) = amrex::TagBox::SET;
            }
        });
    }
}

void AMRContainer::MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm) {
    state[lev].define(ba, dm, nvar, ngrow);

    std::cout << "Compiling initDataFunction from initDataExprParser with AMREX_SPACEDIM = " << AMREX_SPACEDIM
              << " and variable count = " << initDataExprParser.symbols().size() << std::endl;
    amrex::ParserExecutor<AMREX_SPACEDIM> initDataFunction = initDataExprParser.compile<AMREX_SPACEDIM>();

    for (amrex::MFIter mfi(state[lev]); mfi.isValid(); ++mfi) {
        const auto& box = mfi.validbox();
        auto prob_lo = Geom(lev).ProbLoArray();
        auto dx = Geom(lev).CellSizeArray();

        const auto& arr = state[lev].array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
            amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
            amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

            arr(i, j, k, 0) = initDataFunction(x, y, z);
        });
    }
}

void AMRContainer::MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm) {
    state[lev].define(ba, dm, nvar, ngrow);

    amrex::PhysBCFunctNoOp phys_bc;

    // Conservative cell-centered interpolation from lev-1 -> lev
    amrex::InterpFromCoarseLevel(state[lev], time, state[lev - 1], 0, 0, nvar, Geom(lev - 1), Geom(lev), phys_bc, 0,
                                 phys_bc, 0, refRatio(lev - 1), &amrex::cell_cons_interp, bcs, 0);
}

void AMRContainer::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm) {
    // 1. Move old state instance into a temporary local MultiFab
    amrex::MultiFab old_state;
    std::swap(old_state, state[lev]);

    // 2. Allocate state[lev] on the new grid layout and pre-fill newly exposed cells from coarse level
    MakeNewLevelFromCoarse(lev, time, ba, dm);

    // 3. Overwrite overlapping regions with exact data saved in old_state
    amrex::Copy(state[lev], old_state, 0, 0, nvar, 0);

    // 4. Update ghost boundaries across all newly positioned patches
    FillPatch(lev, time);
}

void AMRContainer::ClearLevel(int lev) { state[lev].clear(); }