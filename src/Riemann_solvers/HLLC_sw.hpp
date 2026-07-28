#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/gh_shallow_water.hpp"

class MUSCL_HLLC_sw : public gh_shallow_water
{


public:
    MUSCL_HLLC_sw(SurfaceMesh mesh, std::vector<StateVec> U_in, double gam, size_t threads)
        :  gh_shallow_water(mesh, U_in, gam, threads){}


protected:

    StateVec flux_star(StateVec& u_L, StateVec& u_R, int n_face, int n_edge)
{
    constexpr double GM_local = 0.217909;
    StateVec F_L, F_R, F_L_star, F_R_star,F ,D;
    std::array<double, 2> c_vel;
    double S_star = 0.0;
    double p_L = 0.0, p_R = 0.0;
    double h_L = std::max(u_L[0], density_floor), h_R = std::max(u_R[0], density_floor);
    double u_n_L = 0.0, u_n_R = 0.0;
    double S_R = 0.0, S_L = 0.0;
    vector3d<double> vel_L, vel_R, l_vec, edge_center, nxR;

    int n_edge_1 = n_edge + 1;
    if ((n_edge_1) == faces[n_face].size())
        n_edge_1 = 0;

    edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
    edge_center /= edge_center.norm();

    l_vec[0] = u_L[1]; l_vec[1] = u_L[2]; l_vec[2] = u_L[3];
    vel_L = cross_product(edge_center, l_vec);
    vel_L /= (-std::max(h_L, 1e-14)) * edge_center.norm();

    l_vec[0] = u_R[1]; l_vec[1] = u_R[2]; l_vec[2] = u_R[3];
    vel_R = cross_product(edge_center, l_vec);
    vel_R /= (-std::max(h_R, 1e-14)) * edge_center.norm();

    c_vel = char_vel(u_L, u_R, n_face, n_edge);
    S_L = c_vel[0];
    S_R = c_vel[1];

    F_L = flux(u_L, n_face, n_edge);
    F_R = flux(u_R, n_face, n_edge);

    p_L = 0.5 * GM_local * h_L * h_L;
    p_R = 0.5 * GM_local * h_R * h_R;

    const vector3d<double>& n_hat = edge_normals[n_face][n_edge];
    u_n_L = dot_product(n_hat, vel_L);
    u_n_R = dot_product(n_hat, vel_R);

    double denominator = h_L * (S_L - u_n_L) - h_R * (S_R - u_n_R);
    if (std::abs(denominator) > 1e-14)
    {
        S_star = (p_R - p_L + h_L * u_n_L * (S_L - u_n_L) - h_R * u_n_R * (S_R - u_n_R)) /
                 denominator;
    }
    else
    {
        S_star = 0.5 * (u_n_L + u_n_R);
    }

    if (std::isnan(S_star) || std::isinf(S_star))
        S_star = 0.5 * (u_n_L + u_n_R);

    double P_LR = (p_L + p_R + u_L[0] * (S_L - dot_product(edge_normals[n_face][n_edge], vel_L)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_L)) + u_R[0] * (S_R - dot_product(edge_normals[n_face][n_edge], vel_R)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_R))) / 2;
   // double P_LR = (p_L + p_R + u_L[0] * (S_L - vel_L.norm()) * (S_star -  vel_L.norm()) + u_R[0] * (S_R - vel_R.norm()) * (S_star - vel_R.norm())) / 2;

    nxR = cross_product(edge_normals[n_face][n_edge], (edge_center / edge_center.norm()));

    D[0] = 0;
    D[1] = -nxR[0];
    D[2] = -nxR[1];
    D[3] = -nxR[2];
    D[4] = 0;


    for (size_t i = 0; i < DIM; i++)
    {
        F_L_star[i] = (S_star * (S_L * u_L[i] - F_L[i]) + S_L * P_LR * D[i]) / (S_L - S_star);
        F_R_star[i] = (S_star * (S_R * u_R[i] - F_R[i]) + S_R * P_LR * D[i]) / (S_R - S_star);
    }

    if (S_L >= 0)
    {
        F = F_L;
    }
    else if (S_L < 0 && S_star >= 0)
    {
        F = F_L_star;
    }
    else if (S_star < 0 && S_R >= 0)
    {
        F = F_R_star;
    }
    else if (S_R < 0)
    {
        F = F_R;
    }
    else
    {
        F = F_R;
        std::cout << "flux_star: check char vel, S_R=  " << S_R << " S_L= " << S_L << std::endl;
        stop_check = true;
    }





    // // Fixed: decompose momentum into normal + tangential, replace normal with S_*
    // auto build_star_flux = [&](const StateVec& U_in, const StateVec& F_in,
    //                             double S_K, double u_n_K,
    //                             const vector3d<double>& vel_K,
    //                             StateVec& F_star)
    // {
    //     double h_K   = U_in[0];
    //     double alpha = (S_K - u_n_K) / (S_K - S_star);
    //     double h_star = h_K * alpha;

    //     // tangential velocity = vel - u_n * n_hat
    //     // normal component replaced by S_* in the star region
    //     vector3d<double> vel_t{};
    //     vel_t[0] = vel_K[0] - u_n_K * n_hat[0];
    //     vel_t[1] = vel_K[1] - u_n_K * n_hat[1];
    //     vel_t[2] = vel_K[2] - u_n_K * n_hat[2];

    //     vector3d<double> vel_star{};
    //     vel_star[0] = vel_t[0] + S_star * n_hat[0];
    //     vel_star[1] = vel_t[1] + S_star * n_hat[1];
    //     vel_star[2] = vel_t[2] + S_star * n_hat[2];

    //     StateVec U_star{};
    //     U_star[0] = h_star;
    //     U_star[1] = h_star * vel_star[0];
    //     U_star[2] = h_star * vel_star[1];
    //     U_star[3] = h_star * vel_star[2];
    //     // tracer: c = hc/h, conserved quantity is h*c
    //     double c_K = (h_K > 1e-14) ? (U_in[4] / h_K) : 0.0;
    //     U_star[4] = h_star * c_K;

    //     for (size_t i = 0; i < 5; i++)
    //         F_star[i] = F_in[i] + S_K * (U_star[i] - U_in[i]);
    // };

    // build_star_flux(u_L, F_L, S_L, u_n_L, vel_L, F_L_star);
    // build_star_flux(u_R, F_R, S_R, u_n_R, vel_R, F_R_star);

    // if (S_L >= 0)
    //     F = F_L;
    // else if (S_L < 0 && S_star >= 0)
    //     F = F_L_star;
    // else if (S_star < 0 && S_R >= 0)
    //     F = F_R_star;
    // else if (S_R < 0)
    //     F = F_R;
    // else
    // {
    //     F = F_R;
    //     std::cout << "flux_star: check char vel, S_R= " << S_R << " S_L= " << S_L << std::endl;
    //     stop_check = true;
    // }

     return F;






}




 
};
