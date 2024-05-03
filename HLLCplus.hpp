#pragma once
#include "MUSCL_base.hpp"
#include "MUSCL_geometry.hpp"

class MUSCL_HLLCplus : public MUSCL_base
{

protected:
    std::ofstream outfile, outfile_curl, outfile_p, outfile_omega;
    double omega_ns;

public:
    MUSCL_HLLCplus(SurfaceMesh mesh, std::vector<std::vector<double>> U_in, int dim, double gam)
        : MUSCL_base(mesh, U_in, dim, gam)
    {

        if (dim != 5)
        {
            std::cout << "check dim \n";
            stop_check = true;
        }

        omega_ns = 2;

        outfile.open("results/rho.dat", std::ios::out | std::ios::trunc);
        outfile.close();
        outfile.open("results/rho.dat", std::ios::out | std::ios::app);

        outfile_curl.open("results/curl.dat", std::ios::out | std::ios::trunc);
        outfile_curl.close();
        outfile_curl.open("results/curl.dat", std::ios::out | std::ios::app);

        outfile_p.open("results/p.dat", std::ios::out | std::ios::trunc);
        outfile_p.close();
        outfile_p.open("results/p.dat", std::ios::out | std::ios::app);

        outfile_omega.open("results/omega.dat", std::ios::out | std::ios::trunc);
        outfile_omega.close();
        outfile_omega.open("results/omega.dat", std::ios::out | std::ios::app);
    }

    void print_rho()
    {
        for (auto U_i : U)
        {
            std::cout << U_i[0] << std::endl;
        }
    };

    void write_t_rho()
    {
        outfile << this->time() << "  ";
        for (auto U_i : U)
        {

            outfile << U_i[0] << " ";
        }
        outfile << "\n";
    };
    void write_t_p()
    {
        vector3d<double> vel, l_vec, edge_center;
        double pres;
        outfile_p << this->time() << "  ";
        for (size_t n_face = 0; n_face < faces.size(); n_face++)
        {

            /*l_vec[0] = U[n_face][1];
            l_vec[1] = U[n_face][2];
            l_vec[2] = U[n_face][3];

            vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
            vel /= (-U[n_face][0]);*/

            double test = 0;
            vector3d<double> test1;
            test1[0] = 0;
            test1[1] = 0;
            test1[2] = 0;

            for (size_t n_edge = 0; n_edge < faces[n_face].size(); n_edge++)
            {
                int n_edge_1 = n_edge + 1;
                if ((n_edge_1) == faces[n_face].size())
                {
                    n_edge_1 = 0;
                }

                edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
                edge_center /= edge_center.norm();

                l_vec[0] = U[n_face][1];
                l_vec[1] = U[n_face][2];
                l_vec[2] = U[n_face][3];

                vel = cross_product(edge_center, l_vec);
                vel /= (-U[n_face][0]) * edge_center.norm();

                //test += dot_product(vel, edge_normals[n_face][n_edge]);
                //test1+=cross_product(edge_center, edge_normals[n_face][n_edge])*
                //(vertices[faces[n_face][n_edge]]-vertices[faces[n_face][n_edge_1]]).norm();
            }
           //std::cout << test1.norm() << "\n";

            pres = pressure(U[n_face], vel, face_centers[n_face]);
            outfile_p << pres << " ";
        }
        outfile_p << "\n";
    };

    void write_t_curl()
    {
        vector3d<double> vel, l_vec, rxV, r, edge_center;
        double vort;
        outfile_curl << this->time() << "  ";
        size_t n_edge_1;
        for (size_t n_face = 0; n_face < this->n_faces(); n_face++)
        {

            // vel = cross_product(face_centers[n_face]/face_centers[n_face].norm(), l_vec);
            // vel /= (-U[n_face][0]);

            vort = 0;
            for (size_t n_edge = 0; n_edge < faces[n_face].size(); n_edge++)
            {
                l_vec[0] = U_plus[n_face][n_edge][1];
                l_vec[1] = U_plus[n_face][n_edge][2];
                l_vec[2] = U_plus[n_face][n_edge][3];

                n_edge_1 = n_edge + 1;
                if (n_edge == faces[n_face].size() - 1)
                    n_edge_1 = 0;

                edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
                edge_center /= edge_center.norm();

                vel = cross_product(edge_center, l_vec);
                vel /= (-U[n_face][0]);

                r = (vertices[faces[n_face][n_edge]] - vertices[faces[n_face][n_edge_1]]);
                vort += dot_product(vel, r);
            }

            // rxV = cross_product(face_centers[n_face], vel);
            // outfile_curl << rxV.norm() << " ";

            outfile_curl << vort / surface_area[n_face] << " ";
        }
        outfile_curl << "\n";
    };

    void write_t_omega_z()
    {
        vector3d<double> vel, l_vec, rxV;
        outfile_omega << this->time() << "  ";
        size_t n_edge_1;
        double theta;
        for (size_t n_face = 0; n_face < this->n_faces(); n_face++)
        {
            theta = std::acos(face_centers[n_face][2] / face_centers[n_face].norm());
            l_vec[0] = U[n_face][1];
            l_vec[1] = U[n_face][2];
            l_vec[2] = U[n_face][3];
            vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
            vel /= (-U[n_face][0]);

            // rxV = cross_product(face_centers[n_face], vel);
            // outfile_omega << rxV[2] << " ";

            // std::cout<<std::setprecision(9)<<face_centers[n_face][2]<<"\n";
            outfile_omega << (vel.norm() * vel.norm() - omega_ns * omega_ns * std::sin(theta) * std::sin(theta)) / 2 << " ";
        }
        outfile_omega << "\n";
    };

protected:
    // U = {rho, l1, l2, l3, E}
    std::vector<double> flux(std::vector<double> u_in, int n_face, int n_edge)
    {

        std::vector<double> res;
        res.resize(dim);
        double PI, ndv, L, A, R;
        vector3d<double> vel, l_vec, nxR, edge_center;

        // R = face_centers[n_face].norm();

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center /= edge_center.norm();

        l_vec[0] = u_in[1];
        l_vec[1] = u_in[2];
        l_vec[2] = u_in[3];

        vel = cross_product(edge_center, l_vec);
        vel /= (-u_in[0]) * edge_center.norm();

        // PI = (u_in[4] - u_in[0] * vel.norm() * vel.norm() / 2.) * (gam - 1);
        PI = pressure(u_in, vel, edge_center);

        ndv = dot_product(edge_normals[n_face][n_edge], vel);
        nxR = cross_product(edge_normals[n_face][n_edge], (edge_center / edge_center.norm()));

        double theta = std::acos(edge_center[2] / edge_center.norm());

        res[0] = u_in[0] * dot_product(vel, edge_normals[n_face][n_edge]);
        res[1] = (u_in[1] * ndv - nxR[0] * PI);
        res[2] = (u_in[2] * ndv - nxR[1] * PI);
        res[3] = (u_in[3] * ndv - nxR[2] * PI);
        res[4] = (u_in[4] + PI) * dot_product(vel, edge_normals[n_face][n_edge]);

        return res;
    }

    std::vector<double> flux_star(std::vector<double> u_L, std::vector<double> u_R, int n_face, int n_edge)
    { // returns vector F* or G*
        // HLLC flux

        std::vector<double> F_L, F_R, F_L_star, F_R_star, c_vel, D, F, flux_fix_R, flux_fix_L;
        double S_star, p_L, p_R, rho_R, rho_L, v_L, v_R;
        double S_R, S_L, R;
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

        S_star = (p_R - p_L + rho_L * dot_product(edge_normals[n_face][n_edge], vel_L) * (S_L - dot_product(edge_normals[n_face][n_edge], vel_L)) -
                  rho_R * dot_product(edge_normals[n_face][n_edge], vel_R) *
                      (S_R - dot_product(edge_normals[n_face][n_edge], vel_R))) /
                 (rho_L * (S_L - dot_product(edge_normals[n_face][n_edge], vel_L)) - rho_R * (S_R - dot_product(edge_normals[n_face][n_edge], vel_R)));
        D[4] = S_star;

        double P_LR = (p_L + p_R + u_L[0] * (S_L - dot_product(edge_normals[n_face][n_edge], vel_L)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_L)) + u_R[0] * (S_R - dot_product(edge_normals[n_face][n_edge], vel_R)) * (S_star - dot_product(edge_normals[n_face][n_edge], vel_R))) / 2;

        double V_L=dot_product(edge_normals[n_face][n_edge], vel_L);
        double V_R=dot_product(edge_normals[n_face][n_edge], vel_R);
        double phi_L=u_L[0]*(S_L-V_L); double phi_R=u_R[0]*(S_R-V_R); 

        double a_L = std::sqrt(gam * p_L / u_L[0]);
        double a_R = std::sqrt(gam * p_R / u_R[0]);

        double M=std::min(1., std::max(vel_L.norm()/a_L,vel_R.norm()/a_R));

        int lambd_L=0, lambd_R=0;

        if((V_L-a_L)>0 && (V_R-a_R)<0 )
        lambd_L=1;

        if((V_L+a_L)>0 && (V_R+a_R)<0 )
        lambd_L=1;

        if((V_R-a_R)>0 && (V_L-a_L)<0 )
        lambd_R=1;

        if((V_R+a_R)>0 && (V_L+a_L)<0 )
        lambd_R=1;


        double f_star=0;
        if(lambd_L==0 && lambd_R==0)
        f_star=M*std::sqrt(4+(1-M*M)*(1-M*M))/((1+M*M));
        
        int neighboor_num = neighbors_edge[n_face][n_edge];
        double h=std::min(p_L/p_R,p_R/p_L); 
        //here sould be a loop over all neighboors of both L and R
        //however, the pressure inside the cell != pressure on the edge
        double g=1-pow(h,M);

        vector3d<double> vel_flux_fix_L, angm_flux_fix_L,vel_flux_fix_R, angm_flux_fix_R;

        vel_flux_fix_L[0]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][0]+S_L/(S_L-S_star)*g*(vel_R[0]-vel_L[0]-(V_R-V_L)*edge_normals[n_face][n_edge][0]);
        vel_flux_fix_L[1]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][1]+S_L/(S_L-S_star)*g*(vel_R[1]-vel_L[1]-(V_R-V_L)*edge_normals[n_face][n_edge][1]);
        vel_flux_fix_L[2]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][2]+S_L/(S_L-S_star)*g*(vel_R[2]-vel_L[2]-(V_R-V_L)*edge_normals[n_face][n_edge][2]);

        angm_flux_fix_L=cross_product(edge_center,vel_flux_fix_L);

        vel_flux_fix_R[0]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][0]+S_R/(S_R-S_star)*g*(vel_R[0]-vel_L[0]-(V_R-V_L)*edge_normals[n_face][n_edge][0]);
        vel_flux_fix_R[1]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][1]+S_R/(S_R-S_star)*g*(vel_R[1]-vel_L[1]-(V_R-V_L)*edge_normals[n_face][n_edge][1]);
        vel_flux_fix_R[2]=(f_star-1)*(V_R-V_L)*edge_normals[n_face][n_edge][2]+S_R/(S_R-S_star)*g*(vel_R[2]-vel_L[2]-(V_R-V_L)*edge_normals[n_face][n_edge][2]);

        angm_flux_fix_R=cross_product(edge_center,vel_flux_fix_R);

        
        flux_fix_L[0]=0; flux_fix_R[0]=0;
        flux_fix_L[1]=angm_flux_fix_L[0]*phi_L*phi_R/(phi_R-phi_L); flux_fix_R[1]=angm_flux_fix_R[0]*phi_L*phi_R/(phi_R-phi_L);
        flux_fix_L[2]=angm_flux_fix_L[1]*phi_L*phi_R/(phi_R-phi_L); flux_fix_R[2]=angm_flux_fix_R[1]*phi_L*phi_R/(phi_R-phi_L);
        flux_fix_L[3]=angm_flux_fix_L[2]*phi_L*phi_R/(phi_R-phi_L); flux_fix_R[3]=angm_flux_fix_R[2]*phi_L*phi_R/(phi_R-phi_L);
        flux_fix_L[4]=(f_star-1)*(V_R-V_L)*S_star*phi_R/(phi_R-phi_L); flux_fix_R[4]=(f_star-1)*(V_R-V_L)*S_star*phi_R/(phi_R-phi_L);



        for (size_t i = 0; i < dim; i++)
        {
            F_L_star[i] = (S_star * (S_L * u_L[i] - F_L[i]) + S_L * P_LR * D[i]) / (S_L - S_star)+flux_fix_L[i];
            F_R_star[i] = (S_star * (S_R * u_R[i] - F_R[i]) + S_R * P_LR * D[i]) / (S_R - S_star)+flux_fix_R[i];
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
            std::cout << "flux_star: check char vel, S_R=  " << S_R << " S_L= " << S_L << std::endl;
            stop_check = true;
        }

        return F;
    };

    std::vector<double> source(std::vector<double> u, int n_face)
    { // du/dt

        std::vector<double> res;
        res.resize(dim);
        vector3d<double> edge_center, l_vec, vel, RxV;

        for (size_t i = 0; i < dim; i++)
            res[i] = 0;

        l_vec[0] = u[1];
        l_vec[1] = u[2];
        l_vec[2] = u[3];

        vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
        vel /= (-u[0]);

 

        // compressed star test
        double theta = std::acos(face_centers[n_face][2] / face_centers[n_face].norm());

        double phi = std::atan2(face_centers[n_face][1] / face_centers[n_face].norm(), face_centers[n_face][0] / face_centers[n_face].norm());
        res[1] = -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * (-std::sin(phi)); // x
        res[2] = -omega_ns * omega_ns * u[0] * std::cos(theta) * std::sin(theta) * std::cos(phi);    // y

        //res[4]=-omega_ns*omega_ns*std::sin(theta)*std::cos(theta)*(-std::sin(phi)*l_vec[0]+std::cos(phi)*l_vec[1]); //v2

        return res;
    };

    double pressure(std::vector<double> u, vector3d<double> vel, vector3d<double> r)
    {

        double theta = std::acos(r[2] / r.norm());

        //return  (u[4] - u[0] * vel.norm() * vel.norm() / 2) * (gam - 1); //v1 = uncompressed
        //return  (u[4] - u[0] * vel.norm() * vel.norm() / 2) * (gam - 1) / gam; //v2 = different P
        return (u[4] - u[0] * (vel.norm() * vel.norm() - omega_ns * omega_ns * std::sin(theta) * std::sin(theta)) / 2) * (gam - 1) / gam; // v3 = compressed star + sin
    }

    std::vector<double> char_vel(std::vector<double> u_L, std::vector<double> u_R, int n_face, int n_edge)
    {
        // returns vector {S_L, S_R}
        std::vector<double> res;
        double a_L, a_R, S_L, S_R, p_L, p_R, q_R, q_L;
        vector3d<double> vel_r, vec_r, vel_l, vec_l, edge_center_l, edge_center_r;

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center_r = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center_l = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;

        edge_center_r /= edge_center_r.norm();
        edge_center_l /= edge_center_l.norm();

        vec_l[0] = u_L[1];
        vec_l[1] = u_L[2];
        vec_l[2] = u_L[3];

        vec_r[0] = u_R[1];
        vec_r[1] = u_R[2];
        vec_r[2] = u_R[3];

        vel_l = cross_product(edge_center_l, vec_l);
        vel_l /= (-u_L[0]) * edge_center_l.norm();

        vel_r = cross_product(edge_center_r, vec_r);
        vel_r /= (-u_R[0]) * edge_center_r.norm();

        // p_L = (u_L[4] - u_L[0] * vel_l.norm() * vel_l.norm() / 2) * (gam - 1);
        // p_R = (u_R[4] - u_R[0] * vel_r.norm() * vel_r.norm() / 2) * (gam - 1);

        p_L = pressure(u_L, vel_l, edge_center_l);
        p_R = pressure(u_R, vel_r, edge_center_r);

        a_L = std::sqrt(gam * p_L / u_L[0]);
        a_R = std::sqrt(gam * p_R / u_R[0]);

        double z = (gam - 1) / (2 * gam);
        // double p_star=std::pow((a_L+a_R-(gam-1)/2. * (dot_product(edge_normals[n_face][n_edge], vel_r)-dot_product(edge_normals[n_face][n_edge], vel_l)))
        //  /( a_L/std::pow(p_L,z) + a_R/std::pow(p_R,z) ),1./z);

        double p_pvrs = 1 / 2. * (p_L + p_R) - 1 / 8. * (dot_product(edge_normals[n_face][n_edge], vel_r) - dot_product(edge_normals[n_face][n_edge], vel_l)) * (u_L[0] + u_R[0]) * (a_L + a_R);
        double p_star = std::max(0., p_pvrs);

        if (p_star <= p_R)
        {
            q_R = 1;
        }
        else
        {
            q_R = std::sqrt(1 + (gam + 1) / (2 * gam) * (p_star / p_R - 1));
        }

        if (p_star <= p_L)
        {
            q_L = 1;
        }
        else
        {
            q_L = std::sqrt(1 + (gam + 1) / (2 * gam) * (p_star / p_L - 1));
        }

        S_L = dot_product(vel_l, edge_normals[n_face][n_edge]) - a_L * q_L;
        S_R = dot_product(vel_r, edge_normals[n_face][n_edge]) + a_R * q_R;

        // S_L = std::min(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) - std::max(a_L, a_R);
        // S_R = std::max(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) + std::max(a_L, a_R);

        if (std::isnan(S_L) || std::isnan(S_R))
        {
            std::cout << "p_r: " << p_R << " p_l: " << p_L << std::endl;
            std::cout << "rho_l: " << u_L[0] << " rho_r: " << u_R[0] << std::endl;
            std::cout << std::endl;
            stop_check = true;
        }

        res.resize(2);
        res[0] = S_L;
        res[1] = S_R;

        return res;
    }

    

    std::vector<double> limiter(std::vector<double> u_r, int n_face, int n_edge)
    { // classical Superbee limiter for irregular grids
        // CFL independent
        double etha_minus, etha_plus;
        vector3d<double> R_vec, l_vec, vel, edge_center;
        double R, c, nu_plus;
        std::vector<double> res;
        res.resize(4);

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }
        edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center /= edge_center.norm();

        l_vec[0] = U[n_face][1];
        l_vec[1] = U[n_face][2];
        l_vec[2] = U[n_face][3];

        vel = cross_product(edge_center, l_vec);
        vel /= (-U[n_face][0]) * edge_center.norm();

        // double p = (U[n_face][4] - U[n_face][0] * vel.norm() * vel.norm() / 2) * (gam - 1);
        double p = pressure(U[n_face], vel, edge_center);
        c = std::sqrt(gam * p / U[n_face][0]);

        nu_plus = (c + dot_product(vel, edge_normals[n_face][n_edge])) * dt *
                  (distance(vertices[faces[n_face][n_edge]], vertices[faces[n_face][n_edge_1]]) / surface_area[n_face]);

        etha_plus = H_plus[n_face][n_edge] / BM_dist[n_face][n_edge];
        etha_plus = H_minus[n_face][n_edge] / BM_dist[n_face][n_edge];

        for (size_t i = 0; i < dim; i++)
        {
            res[i] = std::max(0.,
                              std::max(std::min(1., etha_minus * u_r[i] * 2 / (2 * faces[n_face].size() * nu_plus)),
                                       std::min(u_r[i], etha_plus)));

            if (std::isnan(u_r[i]))
            {
                res[i] = 0;
            }
        }

        return res;
    };
};
