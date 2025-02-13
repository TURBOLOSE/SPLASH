#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class MUSCL_HLLCplus : public adiabatic
{


public:
    MUSCL_HLLCplus(SurfaceMesh mesh, std::vector<std::vector<double>> U_in,
     int dim, double gam, double omega_ns_i, bool accretion_on_i, size_t threads)
        :adiabatic(mesh, U_in, dim, gam, omega_ns_i,accretion_on_i, threads){}

protected:
    
    std::vector<double> flux_star(std::vector<double> u_L, std::vector<double> u_R, int n_face, int n_edge)
    {   // HLLC+ flux

        std::vector<double> F_L, F_R, F_L_star, F_R_star, c_vel, D, F, flux_fix_R, flux_fix_L;
        double S_star, p_L, p_R, rho_R, rho_L, v_L, v_R;
        double S_R, S_L, R;
        F_L.resize(dim);
        F_R.resize(dim);
        F_L_star.resize(dim);
        F_R_star.resize(dim);
        D.resize(dim);
        flux_fix_R.resize(dim);
        flux_fix_L.resize(dim);

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
        D[1] = -nxR[0];
        D[2] = -nxR[1];
        D[3] = -nxR[2];

        F_L = flux(u_L, n_face, n_edge);
        F_R = flux(u_R, n_face, n_edge);

        rho_R = u_R[0];
        rho_L = u_L[0];
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

        double phi_L = u_L[0] * (S_L - V_L);
        double phi_R = u_R[0] * (S_R - V_R);

        double gam_L=gam;
        double gam_R=gam;
        

        if(var_gamma){
            gam_L=make_gam(u_L,edge_center);
            gam_R=make_gam(u_R,edge_center);
        }

        double a_L = std::sqrt(gam_L * p_L / u_L[0]);
        double a_R = std::sqrt(gam_R * p_R / u_R[0]);

        double M = std::min(1., std::max(vel_L.norm() / a_L, vel_R.norm() / a_R));

        int lambd_L = 0, lambd_R = 0;

        if ((V_L - a_L) > 0 && (V_R - a_R) < 0)
            lambd_L = 1;

        if ((V_L + a_L) > 0 && (V_R + a_R) < 0)
            lambd_L = 1;

        if ((V_R - a_R) > 0 && (V_L - a_L) < 0)
            lambd_R = 1;

        if ((V_R + a_R) > 0 && (V_L + a_L) < 0)
            lambd_R = 1;

        double f_star = 1;
        if (lambd_L == 0 && lambd_R == 0)
            f_star = M * std::sqrt(4 + (1 - M * M) * (1 - M * M)) / ((1 + M * M));

        int neighboor_num = neighbors_edge[n_face][n_edge];
        double h = std::min(p_L / p_R, p_R / p_L);
        // here sould be a loop over all neighboors of both L and R
        // however, the pressure inside the cell != pressure on the edge
        double g = 1 - pow(h, M);

        vector3d<double> vel_flux_fix_L, angm_flux_fix_L, vel_flux_fix_R, angm_flux_fix_R;

        vel_flux_fix_L[0] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][0] + S_L / (S_L - S_star) * g * (vel_R[0] - vel_L[0] - (V_R - V_L) * edge_normals[n_face][n_edge][0]);
        vel_flux_fix_L[1] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][1] + S_L / (S_L - S_star) * g * (vel_R[1] - vel_L[1] - (V_R - V_L) * edge_normals[n_face][n_edge][1]);
        vel_flux_fix_L[2] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][2] + S_L / (S_L - S_star) * g * (vel_R[2] - vel_L[2] - (V_R - V_L) * edge_normals[n_face][n_edge][2]);

        angm_flux_fix_L = cross_product(edge_center, vel_flux_fix_L);

        vel_flux_fix_R[0] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][0] + S_R / (S_R - S_star) * g * (vel_R[0] - vel_L[0] - (V_R - V_L) * edge_normals[n_face][n_edge][0]);
        vel_flux_fix_R[1] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][1] + S_R / (S_R - S_star) * g * (vel_R[1] - vel_L[1] - (V_R - V_L) * edge_normals[n_face][n_edge][1]);
        vel_flux_fix_R[2] = (f_star - 1) * (V_R - V_L) * edge_normals[n_face][n_edge][2] + S_R / (S_R - S_star) * g * (vel_R[2] - vel_L[2] - (V_R - V_L) * edge_normals[n_face][n_edge][2]);

        angm_flux_fix_R = cross_product(edge_center, vel_flux_fix_R);

        flux_fix_L[0] = 0;
        flux_fix_R[0] = 0;
        flux_fix_L[1] = angm_flux_fix_L[0] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_R[1] = angm_flux_fix_R[0] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_L[2] = angm_flux_fix_L[1] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_R[2] = angm_flux_fix_R[1] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_L[3] = angm_flux_fix_L[2] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_R[3] = angm_flux_fix_R[2] * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_L[4] = (f_star - 1) * (V_R - V_L) * S_star * phi_L * phi_R / (phi_R - phi_L);
        flux_fix_R[4] = (f_star - 1) * (V_R - V_L) * S_star * phi_L * phi_R / (phi_R - phi_L);

        for (size_t i = 0; i < dim; i++)
        {
            F_L_star[i] = (S_star * (S_L * u_L[i] - F_L[i]) + S_L * P_LR * D[i]) / (S_L - S_star) + flux_fix_L[i];
            F_R_star[i] = (S_star * (S_R * u_R[i] - F_R[i]) + S_R * P_LR * D[i]) / (S_R - S_star) + flux_fix_R[i];
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
            std::cout << "flux_star: check char vel, S_R=  " << S_R << " S_L= " << S_L<<" S_Star = "<<S_star << "\n";
            stop_check = true;
        }
        
        return F;
    };

};
