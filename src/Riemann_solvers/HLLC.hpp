#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class MUSCL_HLLC : public adiabatic
{


public:
    MUSCL_HLLC(SurfaceMesh mesh, std::vector<StateVec> U_in, double gam, double omega_ns_i, bool accretion_on_i, size_t threads)
        :adiabatic(mesh, U_in, gam, omega_ns_i,accretion_on_i, threads){}


protected:

    StateVec flux_star(StateVec& u_L, StateVec& u_R, int n_face, int n_edge)
    { // returns vector F* or G*
        // HLLC flux

        StateVec F_L, F_R, F_L_star, F_R_star, D, F;
        std::array<double, 2> c_vel;
        double S_star, p_L, p_R, rho_R, rho_L, v_L, v_R;
        double S_R, S_L, R;

        vector3d<double> vel_L, vel_R, l_vec, nxR, edge_center;

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center /= edge_center.norm();

        l_vec[0] = u_L[1];
        l_vec[1] = u_L[2];
        l_vec[2] = u_L[3];

        vel_L = cross_product(edge_center, l_vec);
        vel_L /= (-u_L[0]) * edge_center.norm();

        l_vec[0] = u_R[1];
        l_vec[1] = u_R[2];
        l_vec[2] = u_R[3];

        vel_R = cross_product(edge_center, l_vec);
        vel_R /= (-u_R[0]) * edge_center.norm();

        c_vel = char_vel(u_L, u_R, n_face, n_edge);
        S_L = c_vel[0];
        S_R = c_vel[1];

        nxR = cross_product(edge_normals[n_face][n_edge], (edge_center / edge_center.norm()));

        D[0] = 0;
        /*D[1] = edge_normals[n_face][n_edge][0];
        D[2] = edge_normals[n_face][n_edge][1];
        D[3] = edge_normals[n_face][n_edge][2];*/

        D[1] = -nxR[0];
        D[2] = -nxR[1];
        D[3] = -nxR[2];

        F_L = flux(u_L, n_face, n_edge);
        F_R = flux(u_R, n_face, n_edge);

        rho_R = u_R[0];
        rho_L = u_L[0];
        // p_L = (u_L[4] - ((vel_L.norm() * vel_L.norm()) * u_L[0]) / 2.) * (gam - 1);
        // p_R = (u_R[4] - ((vel_R.norm() * vel_R.norm()) * u_R[0]) / 2.) * (gam - 1);
        p_L = pressure(u_L, vel_L, edge_center);
        p_R = pressure(u_R, vel_R, edge_center);

        double V_L = dot_product(edge_normals[n_face][n_edge], vel_L);
        double V_R = dot_product(edge_normals[n_face][n_edge], vel_R);

        S_star = (p_R - p_L + rho_L * V_L * (S_L - V_L) -rho_R * V_R *(S_R - V_R)) /
                 (rho_L * (S_L - V_L) - rho_R * (S_R - V_R));
        
        if(std::isnan(S_star)||std::isinf(S_star))
        S_star = (V_L+V_R)/2;

        D[4] = S_star;

        double P_LR = (p_L + p_R + u_L[0] * (S_L - dot_product(edge_normals[n_face][n_edge], vel_L)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_L)) + u_R[0] * (S_R - dot_product(edge_normals[n_face][n_edge], vel_R)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_R))) / 2;

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

        return F;
    };



 
};