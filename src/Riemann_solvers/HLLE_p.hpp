#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class MUSCL_HLLE_p : public adiabatic
{


public:
    MUSCL_HLLE_p(SurfaceMesh mesh, std::vector<std::array<double, 5>> U_in, 
    int dim, double gam, double omega_ns_i, bool accretion_on_i, size_t threads)
        :adiabatic(mesh, U_in, dim, gam, omega_ns_i,accretion_on_i, threads){}



protected:

    std::array<double, 5> flux_star(std::array<double, 5> ul, std::array<double, 5> ur, int n_face, int n_edge)
    {

        std::array<double, 5> FL, FR, F;
        std::array<double, 2> c_vel;
        double S_R, S_L;

        c_vel = char_vel(ul, ur, n_face, n_edge);
        S_L = c_vel[0];
        S_R = c_vel[1];

        FL = flux(ul, n_face, n_edge);
        FR = flux(ur, n_face, n_edge);

        if (S_L >= 0)
        {
            F = FL;
        }
        else if (S_L < 0 && S_R > 0)
        {

            for (size_t i = 0; i < dim; i++)
            {
                F[i] = (S_R * FL[i] - S_L * FR[i] + S_R * S_L * (ur[i] - ul[i])) / (S_R - S_L);
            }
        }
        else if (S_R <= 0)
        {
            F = FR;
        }
        else
        {
            F = FR;
            std::cout << "flux_star: check char vel, S_R=  " << S_R << " S_L= " << S_L << std::endl;
            stop_check = true;
        }

        return F;
    };
};
