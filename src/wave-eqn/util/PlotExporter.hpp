#pragma once

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include "int/Integrator.hpp"
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <filesystem>
namespace fs = std::filesystem;

class PlotExporter {
  public:
    PlotExporter(const amrex::MultiFab& data_, const amrex::Geometry& geom_,
                 const amrex::Vector<std::string>& comp_names_, fs::path path_, int export_interval)
        : data(data_), geom(geom_), comp_names(comp_names_), path(std::move(path_)), export_interval(export_interval) {}

    void exportIfNecessary(int step, amrex::Real time) const {
        if (step % export_interval != 0) // only export when step % export_interval == 0
            return;
        amrex::WriteSingleLevelPlotfile(amrex::Concatenate(path.string(), step, 5), data, comp_names, geom, time, step);
    }

  private:
    const amrex::MultiFab& data;
    const amrex::Geometry& geom;
    const amrex::Vector<std::string>& comp_names;
    fs::path path;
    int export_interval;
};