#pragma once

#include "AMReX.H"
#include "AMReX_AmrCore.H"
#include "AMReX_Interpolater.H"
#include "AMReX_MultiFab.H"
#include "AMReX_Parser.H"

/**
 * @brief AMRContainer manages adaptive mesh refinement levels, field storage,
 *        regridding lifecycle callbacks, and boundary synchronizations using AMReX.
 */
class AMRContainer : public amrex::AmrCore {
  public:
    // No default constructor: level data requires runtime allocation bounds
    AMRContainer() = delete;

    /**
     * @param lev0_geom      Initial coarsest level (level 0) geometry setup.
     * @param amr_info       AMR refinement parameters (max_level, ref_ratio, etc.).
     * @param initDataExpr   Mathematical expression defining the function used to initialise data
     * @param initDataCoords The names of the coordinates/variables used in @param initDataExpr
     * @param varNames       Names of the variables per grid cell, nvar is inferred from this
     * @param ngrow          Number of ghost cells needed around valid patch data.
     */
    AMRContainer(const amrex::Geometry& lev0_geom, const amrex::AmrInfo& amr_info, std::string initDataExpr,
                 std::vector<std::string> initDataCoords, amrex::Vector<std::string> compNames, int ngrow);

    /**
     * @brief Helper utility to inspect layout, box counts, and cell counts across active AMR levels.
     * @param displayLimit Max number of boxes to print per level before abbreviating.
     */
    void printContainerInfo(int displayLimit = 5);

    /**
     * @brief In-place ghost cell update helper for level 'state[lev]'.
     */
    void FillPatch(int lev, amrex::Real time);

    /**
     * @brief Fills valid interior data and ghost cells for a given MultiFab 'dst'.
     *        Executes single-level copy for lev=0, or two-level coarse-fine spatial/temporal
     *        interpolation for lev > 0.
     */
    void FillPatch(amrex::MultiFab& dst, int lev, amrex::Real time);

    const amrex::MultiFab& getState(int lev) const;

    void writeMultiLevelPlotFile(const std::string& filename, amrex::Real time, const amrex::Vector<int>& level_steps);

  public:
    // =========================================================================
    // Pure Virtual Overrides for amrex::AmrCore Lifecycle Methods
    // =========================================================================

    /**
     * @brief Flags cells on 'lev' that require refinement based on error criteria.
     *        AMReX invokes this during regridding to determine fine grid placement.
     */
    virtual void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow_arg) override;

    /**
     * @brief Allocates an entirely new refinement level from scratch (e.g., initial setup).
     */
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) override;

    /**
     * @brief Allocates a new fine level by spatial interpolation from the underlying coarse level.
     */
    virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba,
                                        const amrex::DistributionMapping& dm) override;

    /**
     * @brief Re-grids an existing level while preserving existing fine-level data.
     */
    virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm) override;

    /**
     * @brief Destroys and deallocates grid memory for a removed refinement level.
     */
    virtual void ClearLevel(int lev) override;

  private:
    int ncomp; ///< Number of field components per cell
    amrex::Vector<std::string> compNames;
    int ngrow;                            ///< Ghost cell halo layer size
    amrex::Vector<amrex::MultiFab> state; ///< Per-level storage arrays
    amrex::Vector<amrex::BCRec> bcs;      ///< Per-variable boundary condition descriptor

    std::vector<std::string> initDataCoords;
    std::string initDataExpr;
    amrex::Parser initDataExprParser;
};