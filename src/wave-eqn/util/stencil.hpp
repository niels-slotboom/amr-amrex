#pragma once
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>

#include <cstdint>
#include <utility>

/**
 * Defines compile-time resolved derivative stencils (up to third derivatives, arbitrary stencil width).
 * Note: It is assumed that dx = dy = dz, and multiplication of results by 1/dx^k with k the derivative order is UP TO
 * THE USER.
 */
namespace stencil {
// Directions (x=0, y=1, z=2)
enum dir : int { x = 0, y = 1, z = 2 };

// constexpr math (to be evaluated at compile time)
namespace cexprmath {
// n! (64-bit to prevent overflow for n > 12)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t factorial(int n) {
    std::int64_t res = 1;
    for (int i = 2; i <= n; ++i)
        res *= i;
    return res;
}

// n choose k
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t binom(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n / 2)
        k = n - k;

    std::int64_t res = 1;
    for (int i = 1; i <= k; ++i) {
        res = (res * (n - i + 1)) / i;
    }
    return res;
}

// n^k
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t power(std::int64_t n, int k) {
    std::int64_t result = 1;
    while (k > 0) {
        if (k & 1)
            result *= n;
        n *= n;
        k >>= 1;
    }
    return result;
}

// (-1)^k
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int sign(int k) { return (k & 1) ? -1 : 1; }

// H_(n,m) = sum_(k=1)^n 1/k^m
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr double generalised_harmonic_number(int n, int m) {
    double sum = 0.0;
    for (int k = 1; k <= n; k++) {
        sum += 1.0 / static_cast<double>(power(k, m));
    }
    return sum;
}
} // namespace cexprmath

// weights for pure/unmixed derivatives
namespace weights {
// weights for first derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real first(int offset) {
    static_assert(ngrow > 0, "ngrow needs to be at least 1 to evaluate first derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = offset;
    int k_abs = (k > 0) ? k : -k;

    if (offset == 0)
        return 0;

    std::int64_t num = sign(k_abs + 1) * power(factorial(n), 2);
    std::int64_t denom = k_abs * factorial(n - k_abs) * factorial(n + k_abs);
    double w = static_cast<double>(num) / static_cast<double>(denom);
    return static_cast<amrex::Real>((k > 0) ? w : -w);
}

// weights for second derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real second(int offset) {
    static_assert(ngrow > 0, "ngrow needs to be at least 1 to evaluate second derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = (offset > 0) ? offset : -offset;

    if (offset == 0)
        return -2.0 * generalised_harmonic_number(n, 2);

    std::int64_t num = 2 * sign(k + 1) * power(factorial(n), 2);
    std::int64_t denom = k * k * factorial(n - k) * factorial(n + k);
    return static_cast<amrex::Real>(static_cast<double>(num) / static_cast<double>(denom));
}

// weights for third derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real third(int offset) {
    static_assert(ngrow > 1, "ngrow needs to be at least 2 to evaluate third derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = offset;
    int k_abs = (k > 0) ? k : -k;

    if (offset == 0)
        return 0;

    std::int64_t num = 6 * sign(k_abs + 1) * power(factorial(n), 2);
    std::int64_t denom = power(k_abs, 3) * factorial(n - k_abs) * factorial(n + k_abs);
    double w = ((1.0 - static_cast<double>(k_abs * k_abs) * generalised_harmonic_number(n, 2)) *
                static_cast<double>(num) / static_cast<double>(denom));
    return static_cast<amrex::Real>((k > 0) ? w : -w);
}
} // namespace weights

// --- --- --- FUNDAMENTAL DERIVATIVE STENCIL TEMPLATES --- --- ---
template <int ngrow> constexpr bool always_false = false;

template <int ngrow, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real derivative(int i, int j, int k, int comp,
                                                                const amrex::Array4<const amrex::Real>& arr);

// base template for derivative_impl struct
template <int ngrow, int... Dirs> struct derivative_impl {
    static_assert(always_false<ngrow>,
                  "No stencil::derivative implementation available for requested template parameters");
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        return 0;
    }
};

// --- FIRST DERIVATIVES ---
template <int ngrow, int dir> struct derivative_impl<ngrow, dir> {
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        constexpr int i_offset = (dir == stencil::dir::x);
        constexpr int j_offset = (dir == stencil::dir::y);
        constexpr int k_offset = (dir == stencil::dir::z);

        amrex::Real result = 0.0;
#pragma unroll
        for (int s = 1; s <= ngrow; ++s) {
            result += weights::first<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) -
                                                  arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
        }
        return result;
    }
};

// --- SECOND DERIVATIVES ---
template <int ngrow, int dir0, int dir1> struct derivative_impl<ngrow, dir0, dir1> {
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        if constexpr (dir0 == dir1) { // unmixed derivative
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            amrex::Real result = weights::second<ngrow>(0) * arr(i, j, k, comp);
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::second<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) +
                                                       arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
            }
            return result;
        } else {
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            amrex::Real result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::first<ngrow>(s) *
                          (derivative<ngrow, dir1>(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp, arr) -
                           derivative<ngrow, dir1>(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp, arr));
            }
            return result;
        }
    }
};

// --- THIRD DERIVATIVES ---
template <int ngrow, int dir0, int dir1, int dir2> struct derivative_impl<ngrow, dir0, dir1, dir2> {
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real eval(int i, int j, int k, int comp,
                                                                     const amrex::Array4<const amrex::Real>& arr) {
        if constexpr (dir0 == dir1 && dir1 == dir2) { // unmixed derivative
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            amrex::Real result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::third<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) -
                                                      arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
            }
            return result;
        } else if constexpr (dir0 == dir1 || dir0 == dir2 ||
                             dir1 == dir2) { // derivative like xxy, two in the same direction
            constexpr std::pair<int, int> udirs =
                (dir0 == dir1) ? std::pair<int, int>{dir2, dir0} : std::pair<int, int>{dir0, dir1};
            // udirs.first is the single derivative, udirs.second the double derivative
            constexpr int i_offset = (udirs.first == stencil::dir::x);
            constexpr int j_offset = (udirs.first == stencil::dir::y);
            constexpr int k_offset = (udirs.first == stencil::dir::z);
            // hence take a first derivative along udirs.first of derivative<ngrow,udirs.second,udirs.second>(...)
            amrex::Real result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::first<ngrow>(s) *
                          (derivative<ngrow, udirs.second, udirs.second>(i + s * i_offset, j + s * j_offset,
                                                                         k + s * k_offset, comp, arr) -
                           derivative<ngrow, udirs.second, udirs.second>(i - s * i_offset, j - s * j_offset,
                                                                         k - s * k_offset, comp, arr));
            }
            return result;
        } else {
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            amrex::Real result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result +=
                    weights::first<ngrow>(s) *
                    (derivative<ngrow, dir1, dir2>(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp, arr) -
                     derivative<ngrow, dir1, dir2>(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp, arr));
            }
            return result;
        }
    }
};

template <int ngrow, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real derivative(int i, int j, int k, int comp,
                                                                const amrex::Array4<const amrex::Real>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return derivative_impl<ngrow, Dirs...>::eval(i, j, k, comp, arr);
}

// --- --- --- SPECIAL OPERATORS DERIVED FROM BASIC DERIVATIVE --- --- ---

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

// --- --- --- KREISS-OLIGER DISSIPATION TERMS --- --- ---
template <int ngrow, int dir>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real kreiss_oliger(int i, int j, int k, int comp,
                                                                   const amrex::Array4<const amrex::Real>& arr) {
    static_assert(ngrow > 0, "Can only apply Kreiss-Oliger stencil for ngrow >= 1");
    constexpr int i_offset = (dir == stencil::dir::x) ? 1 : 0;
    constexpr int j_offset = (dir == stencil::dir::y) ? 1 : 0;
    constexpr int k_offset = (dir == stencil::dir::z) ? 1 : 0;

    constexpr int r = ngrow;
    constexpr double norm = static_cast<double>(cexprmath::power(2, 2 * ngrow));

    amrex::Real result = -cexprmath::binom(2 * r, r) * arr(i, j, k, comp);
#pragma unroll
    for (int s = 1; s <= ngrow; ++s) {
        result += -cexprmath::sign(s) * cexprmath::binom(2 * r, r + s) *
                  (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) +
                   arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
    }
    return result * norm;
}
}; // namespace stencil