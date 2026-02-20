#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class MUSCL_HLLC : public adiabatic
{


public:
    MUSCL_HLLC(SurfaceMesh mesh, std::vector<std::vector<double>> U_in, 
    int dim, double gam, double omega_ns_i, bool accretion_on_i, size_t threads)
        :adiabatic(mesh, U_in, dim, gam, omega_ns_i,accretion_on_i, threads){}


protected:

    std::vector<double> flux_star(std::vector<double>& u_L, std::vector<double>& u_R, int n_face, int n_edge)
    { // returns vector F* or G*
        // HLLC flux

        std::vector<double> F_L, F_R, F_L_star, F_R_star, c_vel, D, F;
        double S_star, p_L, p_R, rho_R, rho_L, v_L, v_R;
        double S_R, S_L, R;
        F_L_star.resize(dim);
        F_R_star.resize(dim);
        D.resize(dim);

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

        for (size_t i = 0; i < dim; i++)
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


    // std::vector<double> source(std::vector<double> u, int n_face)
    // { // du/dt

    //     std::vector<double> res;
    //     res.resize(dim);
    //     vector3d<double> edge_center, l_vec, vel, RxV;

    //     for (size_t i = 0; i < dim; i++)
    //         res[i] = 0;

    //     l_vec[0] = u[1];
    //     l_vec[1] = u[2];
    //     l_vec[2] = u[3];

    //     vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
    //     vel /= (-u[0]);

    //     /*if(std::abs(face_centers[n_face][2]*std::cos(M_PI/8) +face_centers[n_face][1]*std::sin(M_PI/8))  <0.1){
    //     res[0]=1;
    //     //res[4]=vel.norm()*vel.norm()/2. * res[0]+1;
    //     res[4]=vel.norm()*vel.norm()/2. * res[0];
    //     }*/

    //     /*//visc test
    //     RxV=cross_product(face_centers[n_face]/face_centers[n_face].norm(),vel);
    //     res[1]=-RxV[0];
    //     res[2]=-RxV[1];
    //     res[3]=-RxV[2];*/

    //     /* if(std::abs(face_centers[n_face][2])  <0.1){ //energy dissipation test
    //         double e_here;
    //         vector3d<double> vel_1;
    //         vel_1*=-0.5;
    //         res[0]=1;
    //         RxV=cross_product(face_centers[n_face]/face_centers[n_face].norm(),vel_1);
    //         res[1]=RxV[0]*u[0];
    //         res[2]=RxV[1]*u[0];
    //         res[3]=RxV[2]*u[0];
    //         e_here=vel_1.norm()*vel_1.norm()/2.+1/(gam-1);
    //         res[4]=e_here+(vel-vel_1).norm()*(vel-vel_1).norm()/2;

    //     }*/

    //     // compressed star test
    //     double theta = std::acos(face_centers[n_face][2] / face_centers[n_face].norm());

    //     double phi = std::atan2(face_centers[n_face][1] / face_centers[n_face].norm(), face_centers[n_face][0] / face_centers[n_face].norm());

    //     res[1] = -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * (-std::sin(phi)); // x
    //     res[2] = -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * std::cos(phi);    // y
    //     //res[4] = -omega_ns*omega_ns* u[0] *std::sin(theta)*std::cos(theta)*(-std::sin(phi)*l_vec[0]+std::cos(phi)*l_vec[1]); //v2


    //     /*for (size_t n_edge = 0; n_edge  < faces[n_face].size(); n_edge ++)
    //     {   

    //         double theta = std::acos(vertices[faces[n_face][n_edge]][2] / vertices[faces[n_face][n_edge]].norm());
    //         double phi = std::atan2(vertices[faces[n_face][n_edge]][1] / vertices[faces[n_face][n_edge]].norm(), vertices[faces[n_face][n_edge]][0] / vertices[faces[n_face][n_edge]].norm());
    //         res[1] += -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * (-std::sin(phi)) /faces[n_face].size(); // x
    //         res[2] += -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * std::cos(phi) /faces[n_face].size();    // y
    //     }*/
        


    //     /*double p = pressure(u,vel,face_centers[n_face]);
    //     static double max_val=0;
    //     double val=abs(-omega_ns*omega_ns*u[0] * std::cos(theta) * std::sin(theta)*0.002/std::sqrt(gam*p/u[0]));
    //     if(val > max_val)
    //     max_val = val;
    //     std::cout<<max_val<<"\n";*/

    //     // res[4]=-omega_ns*omega_ns*u[0]*std::sin(theta)*std::cos(theta)*
    //     //(std::cos(theta)*(std::cos(phi)*vel[0]+std::sin(phi)*vel[1])-std::sin(theta)*vel[2]); //v1

    //     // res[4]=-omega_ns*omega_ns*std::sin(theta)*std::cos(theta)*
    //     //(-std::sin(phi)*l_vec[0]+std::cos(phi)*l_vec[1]); //v2

    //     // vector3d<double> g,fc;
    //     // fc=face_centers[n_face]/face_centers[n_face].norm();
    //     /*g[0]=fc[0]*fc[2]/std::sqrt(fc[0]*fc[0]+fc[1]*fc[1])*(-omega_ns*omega_ns*u[0] * std::cos(theta) *std::sin(theta));
    //     g[1]=fc[1]*fc[2]/std::sqrt(fc[0]*fc[0]+fc[1]*fc[1])*(-omega_ns*omega_ns*u[0] * std::cos(theta) *std::sin(theta));
    //     g[2]=-std::sqrt(fc[0]*fc[0]+fc[1]*fc[1])*(-omega_ns*omega_ns*u[0] * std::cos(theta) *std::sin(theta));
    //      res[4]=u[0]*dot_product(g,vel);
    //      */

    //     return res;
    // };

 
};
