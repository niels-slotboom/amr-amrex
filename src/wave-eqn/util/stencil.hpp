#pragma once
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>

// any stencil is meant to be multiplied by 1/dx^k with k the order of the differential operator
// it is assumed that dx = dy = dz, i.e. that the grid spacing its the same in all directions
namespace stencil {

enum dir : int { x = 0, y = 1, z = 2 };

template <int ngrow> inline constexpr bool always_false = false; // needed for delayed static_assert(false)

// --- PARTIAL DERIVATIVES --- (coefficients taken from https://en.wikipedia.org/wiki/Finite_difference_coefficient)

// forward declaration of interface function
template <int ngrow, int... Dirs>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real derivative(int i, int j, int k, int comp,
                                                                const amrex::Array4<const amrex::Real>& arr);

template <int ngrow, int... Dirs> struct derivative_impl {
    static_assert(always_false<ngrow>,
                  "stencil::derivative is not implemented for requested ngrow and/or direction count");
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        return 0.0;
    }
};

// first derivatives
template <int ngrow, int dir0> struct derivative_impl<ngrow, dir0> {
    static_assert(dir0 >= 0 && dir0 < 3, "stencil::derivative directions must be 0,1,2");
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        constexpr int i_offset = (dir0 == 0) ? 1 : 0;
        constexpr int j_offset = (dir0 == 1) ? 1 : 0;
        constexpr int k_offset = (dir0 == 2) ? 1 : 0;

        std::array<amrex::Real, ngrow> diffs;
        diffs[0] =
            arr(i + i_offset, j + j_offset, k + k_offset, comp) - arr(i - i_offset, j - j_offset, k - k_offset, comp);

        if constexpr (ngrow > 1)
            diffs[1] = arr(i + 2 * i_offset, j + 2 * j_offset, k + 2 * k_offset, comp) -
                       arr(i - 2 * i_offset, j - 2 * j_offset, k - 2 * k_offset, comp);
        if constexpr (ngrow > 2)
            diffs[2] = arr(i + 3 * i_offset, j + 3 * j_offset, k + 3 * k_offset, comp) -
                       arr(i - 3 * i_offset, j - 3 * j_offset, k - 3 * k_offset, comp);
        if constexpr (ngrow > 3)
            diffs[3] = arr(i + 4 * i_offset, j + 4 * j_offset, k + 4 * k_offset, comp) -
                       arr(i - 4 * i_offset, j - 4 * j_offset, k - 4 * k_offset, comp);

        if constexpr (ngrow == 1) {
            return (1.0 / 2.0) * diffs[0];
        } else if constexpr (ngrow == 2) {
            return (2.0 / 3.0) * diffs[0] + (-1.0 / 12.0) * diffs[1];
        } else if constexpr (ngrow == 3) {
            return (3.0 / 4.0) * diffs[0] + (-3.0 / 20.0) * diffs[1] + (1.0 / 60.0) * diffs[2];
        } else if constexpr (ngrow == 4) {
            return (4.0 / 5.0) * diffs[0] + (-1.0 / 5.0) * diffs[1] + (4.0 / 105.0) * diffs[2] +
                   (-1.0 / 280.0) * diffs[3];
        } else {
            static_assert(always_false<ngrow>,
                          "stencil::derivative: first derivative not implemented for requested ngrow");
            return 0.0;
        }
    }
};

// second derivatives
template <int ngrow, int dir0, int dir1> struct derivative_impl<ngrow, dir0, dir1> {
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE static amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        constexpr int i_offset = (dir0 == 0) ? 1 : 0;
        constexpr int j_offset = (dir0 == 1) ? 1 : 0;
        constexpr int k_offset = (dir0 == 2) ? 1 : 0;
        if constexpr (dir0 == dir1) { // both derivatives in same direction
            std::array<amrex::Real, ngrow + 1> terms;
            terms[0] = arr(i, j, k, comp);
            terms[1] = arr(i + i_offset, j + j_offset, k + k_offset, comp) +
                       arr(i - i_offset, j - j_offset, k - k_offset, comp);
            if constexpr (ngrow > 1)
                terms[2] = arr(i + 2 * i_offset, j + 2 * j_offset, k + 2 * k_offset, comp) +
                           arr(i - 2 * i_offset, j - 2 * j_offset, k - 2 * k_offset, comp);
            if constexpr (ngrow > 2)
                terms[3] = arr(i + 3 * i_offset, j + 3 * j_offset, k + 3 * k_offset, comp) +
                           arr(i - 3 * i_offset, j - 3 * j_offset, k - 3 * k_offset, comp);
            if constexpr (ngrow > 3)
                terms[4] = arr(i + 4 * i_offset, j + 4 * j_offset, k + 4 * k_offset, comp) +
                           arr(i - 4 * i_offset, j - 4 * j_offset, k - 4 * k_offset, comp);
            if constexpr (ngrow == 1) {
                return -2.0 * terms[0] + terms[1];
            } else if constexpr (ngrow == 2) {
                return (-5.0 / 2.0) * terms[0] + (4.0 / 3.0) * terms[1] + (-1.0 / 12.0) * terms[2];
            } else if constexpr (ngrow == 3) {
                return (-49.0 / 18.0) * terms[0] + (3.0 / 2.0) * terms[1] + (-3.0 / 20.0) * terms[2] +
                       (1.0 / 90.0) * terms[3];
            } else if constexpr (ngrow == 4) {
                return (-205.0 / 72.0) * terms[0] + (8.0 / 5.0) * terms[1] + (-1.0 / 5.0) * terms[2] +
                       (8.0 / 315.0) * terms[3] + (-1.0 / 560.0) * terms[4];
            } else {
                static_assert(always_false<ngrow>,
                              "stencil::derivative: unmixed second derivative not implemented for requested ngrow");
                return 0.0;
            }
        } else { // mixed derivatives
            std::array<amrex::Real, ngrow> terms;
            terms[0] = derivative<ngrow, dir1>(i + i_offset, j + j_offset, k + k_offset, comp, arr) -
                       derivative<ngrow, dir1>(i - i_offset, j - j_offset, k - k_offset, comp, arr);
            if constexpr (ngrow > 1)
                terms[1] = derivative<ngrow, dir1>(i + 2 * i_offset, j + 2 * j_offset, k + 2 * k_offset, comp, arr) -
                           derivative<ngrow, dir1>(i - 2 * i_offset, j - 2 * j_offset, k - 2 * k_offset, comp, arr);
            if constexpr (ngrow > 2)
                terms[2] = derivative<ngrow, dir1>(i + 3 * i_offset, j + 3 * j_offset, k + 3 * k_offset, comp, arr) -
                           derivative<ngrow, dir1>(i - 3 * i_offset, j - 3 * j_offset, k - 3 * k_offset, comp, arr);
            if constexpr (ngrow > 3)
                terms[3] = derivative<ngrow, dir1>(i + 4 * i_offset, j + 4 * j_offset, k + 4 * k_offset, comp, arr) -
                           derivative<ngrow, dir1>(i - 4 * i_offset, j - 4 * j_offset, k - 4 * k_offset, comp, arr);
            if constexpr (ngrow == 1) {
                return (1.0 / 2.0) * terms[0];
            } else if constexpr (ngrow == 2) {
                return (2.0 / 3.0) * terms[0] + (-1.0 / 12.0) * terms[1];
            } else if constexpr (ngrow == 3) {
                return (3.0 / 4.0) * terms[0] + (-3.0 / 20.0) * terms[1] + (1.0 / 60.0) * terms[2];
            } else if constexpr (ngrow == 4) {
                return (4.0 / 5.0) * terms[0] + (-1.0 / 5.0) * terms[1] + (4.0 / 105.0) * terms[2] +
                       (-1.0 / 280.0) * terms[3];
            } else {
                static_assert(always_false<ngrow>,
                              "stencil::derivative: mixed second derivative not implemented for requested ngrow");
                return 0.0;
            }
        }
    };
};

template <int ngrow, int... Dirs>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real derivative(int i, int j, int k, int comp,
                                                                const amrex::Array4<const amrex::Real>& arr) {
    static_assert(ngrow > 0, "stencil::derivative requires ngrow of at least 1");
    return derivative_impl<ngrow, Dirs...>::eval(i, j, k, comp, arr);
}

// --- GRADIENT ---
template <int ngrow>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 3>
gradient(int i, int j, int k, int comp, const amrex::Array4<const amrex::Real>& arr) {
    return {
        derivative<ngrow, dir::x>(i, j, k, comp, arr),
        derivative<ngrow, dir::y>(i, j, k, comp, arr),
        derivative<ngrow, dir::z>(i, j, k, comp, arr),
    };
}

template <int ngrow>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real gradient_squared(int i, int j, int k, int comp,
                                                                      const amrex::Array4<const amrex::Real>& arr) {
    const auto grad = gradient<ngrow>(i, j, k, comp, arr);
    return grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
}

// --- LAPLACIAN ---
// NOTE: This only works for dx = dy = dz.
template <int ngrow>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real laplacian(int i, int j, int k, int comp,
                                                               const amrex::Array4<const amrex::Real>& arr) {
    return derivative<ngrow, dir::x, dir::x>(i, j, k, comp, arr) +
           derivative<ngrow, dir::y, dir::y>(i, j, k, comp, arr) +
           derivative<ngrow, dir::z, dir::z>(i, j, k, comp, arr);
}

// --- HESSIAN ---
template <int ngrow>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Array2D<amrex::Real, 0, 2, 0, 2>
hessian(int i, int j, int k, int comp, const amrex::Array4<const amrex::Real>& arr) {
    amrex::Array2D<amrex::Real, 0, 2, 0, 2> H;
    // diagonal
    H(0, 0) = derivative<ngrow, dir::x, dir::x>(i, j, k, comp, arr);
    H(1, 1) = derivative<ngrow, dir::y, dir::y>(i, j, k, comp, arr);
    H(2, 2) = derivative<ngrow, dir::z, dir::z>(i, j, k, comp, arr);
    // off-diagonal
    H(0, 1) = H(1, 0) = derivative<ngrow, dir::x, dir::y>(i, j, k, comp, arr);
    H(0, 2) = H(2, 0) = derivative<ngrow, dir::x, dir::z>(i, j, k, comp, arr);
    H(1, 2) = H(2, 1) = derivative<ngrow, dir::y, dir::z>(i, j, k, comp, arr);

    return H;
}

// --- KREISS-OLIGER ---
// TODO: Implement
} // namespace stencil