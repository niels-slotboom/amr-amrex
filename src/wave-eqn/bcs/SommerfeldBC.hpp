#pragma once

#include "AMReX_Extension.H"
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_RealVect.H"
#include "rhs/WaveEqnRHS.hpp"

#include "util/stencil.hpp"

// assumes dx = dy = dz
// edge/corner cells of the halo are invalid, but unused by the WaveEqnRHS so dont matter
struct SommerfeldBC {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = 1;

    using Component = WaveEqnRHS<ngrow>::Component;

    AMREX_GPU_HOST_DEVICE
    void operator()(const amrex::IntVect& iv, const amrex::Array4<amrex::Real>& dest, int dcomp, int numcomp,
                    const amrex::GeometryData& geom, amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const {
        using namespace stencil;

        const amrex::Box domain = geom.Domain();

        amrex::IntVect nearest_int = nearest_interior(iv, domain);
        amrex::IntVect normal_vec = nearest_int - iv;

        if (normal_vec == amrex::IntVect{0, 0, 0})
            return;

        // --- --- --- Coefficients --- --- ---
        // Boundary Point Coordinates
        amrex::GpuArray<amrex::Real, 3> bdry_pt_coords = boundary_coordinates(nearest_int, geom);
        amrex::Real r = std::sqrt(bdry_pt_coords[dir::x] * bdry_pt_coords[dir::x] +
                                  bdry_pt_coords[dir::y] * bdry_pt_coords[dir::y] +
                                  bdry_pt_coords[dir::z] * bdry_pt_coords[dir::z]);

        // select normal coordinate
        int normal_dir = (normal_vec[dir::x] != 0) ? dir::x : (normal_vec[dir::y] != 0) ? dir::y : dir::z;
        amrex::Real normal_coord = bdry_pt_coords[normal_dir];
        int epsilon = normal_vec[normal_dir]; // sign of the boundary

        // fix normal_vec to be something valid for edge/corner ghost cells, remains the same if face ghost cell
        normal_vec = amrex::IntVect{0, 0, 0};
        normal_vec[normal_dir] = epsilon;

        // Grid spacing
        amrex::Real dx = geom.CellSize(dir::x);
        amrex::Real dx_inv = 1.0 / dx;

        // coefficients
        amrex::Real alpha = dx / (2 * normal_coord);
        amrex::Real beta = 0.5 * dx;
        amrex::Real beta_inv = 2.0 * dx_inv;
        amrex::Real gamma = r / normal_coord;

        // --- --- --- Stencils --- --- ---
        auto [transverse_1st_derivs_phi, transverse_1st_derivs_pi] =
            transverse_1st_deriv_term(nearest_int, domain, dest, bdry_pt_coords, normal_coord);

        amrex::Real transverse_laplacian = compute_transverse_laplacian(nearest_int, domain, dest);

        amrex::Real normal_2nd_deriv_term = compute_normal_2nd_deriv_term(nearest_int, normal_vec, dest);

        // --- --- --- Assemble final result --- --- ---
        amrex::Real phi_center = dest(nearest_int, Component::phi);
        amrex::Real pi_center = dest(nearest_int, Component::dphi_dt);

        amrex::Real phi_new = (1.0 + (2 * epsilon) * alpha) * phi_center + epsilon * transverse_1st_derivs_phi +
                              (2 * epsilon) * gamma * beta * pi_center;

        amrex::Real pi_new =
            (1.0 + (2 * epsilon) * alpha) * pi_center + epsilon * transverse_1st_derivs_pi +
            epsilon * (0.5 * gamma * beta_inv) * (phi_new + normal_2nd_deriv_term + transverse_laplacian);

        dest(iv, Component::phi) = phi_new;
        dest(iv, Component::dphi_dt) = pi_new;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::IntVect nearest_interior(const amrex::IntVect& iv,
                                                                             const amrex::Box& domain) const {
        using namespace stencil;

        amrex::IntVect result{};
        result[dir::x] = amrex::Clamp(iv[dir::x], domain.smallEnd(dir::x), domain.bigEnd(dir::x));
        result[dir::y] = amrex::Clamp(iv[dir::y], domain.smallEnd(dir::y), domain.bigEnd(dir::y));
        result[dir::z] = amrex::Clamp(iv[dir::z], domain.smallEnd(dir::z), domain.bigEnd(dir::z));
        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<int, 3>
    is_outermost_interior(const amrex::IntVect& idx, const amrex::Box& domain) const {
        using namespace stencil;

        amrex::GpuArray<int, 3> result{0, 0, 0};
        result[dir::x] = (idx[dir::x] == domain.smallEnd(dir::x)) ? -1 : result[dir::x];
        result[dir::x] = (idx[dir::x] == domain.bigEnd(dir::x)) ? 1 : result[dir::x];

        result[dir::y] = (idx[dir::y] == domain.smallEnd(dir::y)) ? -1 : result[dir::y];
        result[dir::y] = (idx[dir::y] == domain.bigEnd(dir::y)) ? 1 : result[dir::y];

        result[dir::z] = (idx[dir::z] == domain.smallEnd(dir::z)) ? -1 : result[dir::z];
        result[dir::z] = (idx[dir::z] == domain.bigEnd(dir::z)) ? 1 : result[dir::z];

        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
    boundary_coordinates(const amrex::IntVect& nearest_int, const amrex::GeometryData& geom) const {
        using namespace stencil;

        const auto domain = geom.Domain();
        const auto is_outermost = is_outermost_interior(nearest_int, domain);
        amrex::GpuArray<amrex::Real, 3> result;
        result[dir::x] =
            geom.ProbLo(dir::x) +
            (nearest_int[dir::x] - domain.smallEnd(dir::x) + 0.5 * (is_outermost[dir::x] + 1)) * geom.CellSize(dir::x);
        result[dir::y] =
            geom.ProbLo(dir::y) +
            (nearest_int[dir::y] - domain.smallEnd(dir::y) + 0.5 * (is_outermost[dir::y] + 1)) * geom.CellSize(dir::y);
        result[dir::z] =
            geom.ProbLo(dir::z) +
            (nearest_int[dir::z] - domain.smallEnd(dir::z) + 0.5 * (is_outermost[dir::z] + 1)) * geom.CellSize(dir::z);

        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    compute_transverse_laplacian(const amrex::IntVect& nearest_int, const amrex::Box& domain,
                                 const amrex::Array4<const amrex::Real>& dest) const {
        using namespace stencil;

        auto is_outermost = is_outermost_interior(nearest_int, domain);

        amrex::Real phi_xx =
            (is_outermost[dir::x] != 0) ? 0.0 : derivative<ngrow, dir::x, dir::x>(nearest_int, Component::phi, dest);
        amrex::Real phi_yy =
            (is_outermost[dir::y] != 0) ? 0.0 : derivative<ngrow, dir::y, dir::y>(nearest_int, Component::phi, dest);
        amrex::Real phi_zz =
            (is_outermost[dir::z] != 0) ? 0.0 : derivative<ngrow, dir::z, dir::z>(nearest_int, Component::phi, dest);

        return phi_xx + phi_yy + phi_zz;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::pair<amrex::Real, amrex::Real>
    transverse_1st_deriv_term(const amrex::IntVect& nearest_int, const amrex::Box& domain,
                              const amrex::Array4<const amrex::Real>& dest, amrex::GpuArray<amrex::Real, 3> coords,
                              amrex::Real normal_coord) const {
        using namespace stencil;

        auto is_outermost = is_outermost_interior(nearest_int, domain);

        std::pair<amrex::Real, amrex::Real> result{0.0, 0.0};

        amrex::Real inv_normal_coord = 1.0 / normal_coord;

        amrex::Real x_fac = coords[dir::x] * inv_normal_coord;
        amrex::Real y_fac = coords[dir::y] * inv_normal_coord;
        amrex::Real z_fac = coords[dir::z] * inv_normal_coord;

        // conditionally computed first phi derivatives, if not computable then zero
        amrex::Real phi_x =
            (is_outermost[dir::x] != 0) ? 0.0 : derivative<ngrow, dir::x>(nearest_int, Component::phi, dest);
        result.first += x_fac * phi_x;

        amrex::Real phi_y =
            (is_outermost[dir::y] != 0) ? 0.0 : derivative<ngrow, dir::y>(nearest_int, Component::phi, dest);
        result.first += y_fac * phi_y;

        amrex::Real phi_z =
            (is_outermost[dir::z] != 0) ? 0.0 : derivative<ngrow, dir::z>(nearest_int, Component::phi, dest);
        result.first += z_fac * phi_z;

        // conditionally computed first pi derivatives, if not computable then zero
        amrex::Real pi_x =
            (is_outermost[dir::x] != 0) ? 0.0 : derivative<ngrow, dir::x>(nearest_int, Component::dphi_dt, dest);
        result.second += x_fac * pi_x;

        amrex::Real pi_y =
            (is_outermost[dir::y] != 0) ? 0.0 : derivative<ngrow, dir::y>(nearest_int, Component::dphi_dt, dest);
        result.second += y_fac * pi_y;

        amrex::Real pi_z =
            (is_outermost[dir::z] != 0) ? 0.0 : derivative<ngrow, dir::z>(nearest_int, Component::dphi_dt, dest);
        result.second += z_fac * pi_z;

        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    compute_normal_2nd_deriv_term(const amrex::IntVect& nearest_int, const amrex::IntVect& normal_vec,
                                  const amrex::Array4<const amrex::Real>& dest) const {
        return -2.0 * dest(nearest_int, Component::phi) + dest(nearest_int + normal_vec, Component::phi);
    }
};