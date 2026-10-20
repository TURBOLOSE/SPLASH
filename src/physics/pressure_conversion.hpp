#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace splash {

// Literal conservative beta closure used by pressure_fc()/make_beta(). This
// deliberately retains the two Newton steps; it does not replace the EOS by
// a converged beta root or by source()'s different pressure-based closure.
inline double conservative_gamma(double sigma, double internal, double speed2,
                                 double fixed_gamma, bool variable_gamma)
{
    if (!variable_gamma) return fixed_gamma;
    const double gravity = .217909-speed2;
    const double C = gravity >= 0. ? 12./5*1.6e-13*sigma/internal*
        std::pow(3./4*4.85e36*gravity*sigma, 1./4) :
        std::numeric_limits<double>::quiet_NaN();
    const double beta_switch = .736194670678821;
    const double C_switch = -1-1/(beta_switch-1);
    const double ceiling = 1-1e-9;
    double beta = C <= C_switch ? 1-1/(1+C) : 1-std::pow(2/C,4);
    if (beta > ceiling || !std::isfinite(beta)) beta = ceiling;
    for (int iteration = 0; iteration < 2; ++iteration)
        beta -= (beta/(std::pow(1-beta,1./4)*(1-beta/2))-C)/
                (-(beta*beta+6*beta-8)/(2*(2-beta)*(2-beta)*std::pow(1-beta,5./4)));
    if (beta < 0.) beta = 0.;
    if (beta > ceiling || !std::isfinite(beta)) beta = ceiling;
    return (10-3*beta)/(8-3*beta);
}

inline double pressure_from_internal(double sigma, double internal, double speed2,
                                     double fixed_gamma, bool variable_gamma)
{
    return internal*(conservative_gamma(sigma,internal,speed2,fixed_gamma,variable_gamma)-1);
}

// Inverse of the implemented conservative pressure map, not of an independent
// pressure-beta approximation. A safeguarded Newton iteration is cheap in the
// radiation dominated regime; bracketed fallback also covers clipped branches.
inline double internal_from_pressure(double sigma, double pressure, double speed2,
                                     double fixed_gamma, bool variable_gamma)
{
    if (!(sigma > 0.) || !(pressure > 0.) || !std::isfinite(sigma) ||
        !std::isfinite(pressure) || !std::isfinite(speed2))
        throw std::invalid_argument("Nonphysical state in pressure-to-energy conversion");
    if (!variable_gamma) return pressure/(fixed_gamma-1);
    double lo = 2.49*pressure, hi = 4.01*pressure;
    double value = .5*(lo+hi);
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double residual = pressure_from_internal(sigma,value,speed2,fixed_gamma,true)-pressure;
        if (std::abs(residual) <= 2e-13*pressure) return value;
        if (residual < 0.) lo = value; else hi = value;
        const double h = 1e-5*value;
        const double derivative = (pressure_from_internal(sigma,value+h,speed2,fixed_gamma,true)-
                                   pressure_from_internal(sigma,value-h,speed2,fixed_gamma,true))/(2*h);
        const double trial = value-residual/derivative;
        value = std::isfinite(trial) && derivative > 0. && trial > lo && trial < hi ?
                trial : .5*(lo+hi);
    }
    throw std::runtime_error("Pressure-to-energy inversion did not converge");
}

} // namespace splash
