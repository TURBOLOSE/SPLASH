#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace splash {

// Radius-one Cartesian angular momentum must lie in the local tangent plane.
// Return the change so callers can account for the numerical torque explicitly.
template <class State, class Vec>
std::array<double, 3> project_angular_momentum(State& u, const Vec& r)
{
    const double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    if (!(r2 > 0.) || !std::isfinite(r2))
        throw std::invalid_argument("Invalid radius in angular-momentum projection");
    const double radial = (u[1]*r[0] + u[2]*r[1] + u[3]*r[2])/r2;
    std::array<double, 3> change{};
    for (std::size_t k = 0; k < 3; ++k) {
        change[k] = -radial*r[k];
        u[k+1] += change[k];
    }
    return change;
}

template <class State, class Vec>
double radial_angular_momentum_squared(const State& u, const Vec& r)
{
    const double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    const double dot = u[1]*r[0] + u[2]*r[1] + u[3]*r[2];
    return dot*dot/r2;
}

// Accepted explicit midpoint, or the implemented iterated trapezoidal update.
// Predictor and unsuccessful fixed-point iterations carry no extra weight.
template <std::size_t N>
std::array<double, N> accepted_source_increment(
    const std::array<double, N>& first, const std::array<double, N>& last,
    double dt, bool trapezoidal)
{
    std::array<double, N> result{};
    for (std::size_t k = 0; k < N; ++k)
        result[k] = dt*(trapezoidal ? .5*(first[k]+last[k]) : last[k]);
    return result;
}

} // namespace splash
