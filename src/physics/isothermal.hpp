#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class isothermal : public MUSCL_base
{

private:
    std::ofstream outfile, outfile_curl, outfile_p, outfile_omega, outfile_l[3], outfile_mach;

public:
    isothermal(SurfaceMesh mesh, std::vector<StateVec> U_in, double gam, size_t threads)
        : MUSCL_base(mesh, U_in, gam,0,threads)
    {

        set_analytical_solution();
        if (DIM != 4)
        {
            std::cout << "check DIM \n";
            stop_check = true;
        }

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

        std::string adrs[] = {"results/Lx.dat", "results/Ly.dat", "results/Lz.dat"};

        for (size_t i=0; i < 3; i++)
        {
            outfile_l[i].open(adrs[i], std::ios::out | std::ios::trunc);
            outfile_l[i].close();
            outfile_l[i].open(adrs[i], std::ios::out | std::ios::app);
        }

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

        void write_t_mach()
    {
        vector3d<double> vel, l_vec, edge_center;
        double pres;
        outfile_mach << this->time() << "  ";
        for (size_t n_face = 0; n_face < faces.size(); n_face++)
        {

            l_vec[0] = U[n_face][1];
            l_vec[1] = U[n_face][2];
            l_vec[2] = U[n_face][3];

            vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
            vel /= (-U[n_face][0]);
            outfile_mach << vel.norm() / a << " ";
        }
        outfile_mach << "\n";
    };
        void write_t_L()
    {
        for (size_t i=0; i < 3; i++)
        {

            outfile_l[i] << this->time() << "  ";
            for (auto U_j : U)
            {
                // outfile_l[i].flush()
                outfile_l[i] << U_j[i + 1] << " ";
                // out_lc<< U_i[0] << " ";
            }
            outfile_l[i] << "\n";
        }
    };

     void write_t_p()
    {
        vector3d<double> vel, l_vec, edge_center;
        double pres;
        outfile_p << this->time() << "  ";
        for (size_t n_face = 0; n_face < faces.size(); n_face++)
        {
            outfile_p << a*a*U[n_face][0] << " ";
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

            rxV = cross_product(face_centers[n_face], vel);
            outfile_omega << rxV[2] << " ";

        }
        outfile_omega << "\n";
    };



public:
    //const double a = 1;


    StateVec flux(StateVec u_in, int n_face, int n_edge)
    {
        StateVec res;
        double PI, ndv, L, A, R;
        vector3d<double> R_vec, vel, vel1, vel2, l_vec, nxR, edge_center, omxv1;

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center /= edge_center.norm(); //unit sphere


        PI = a * a * u_in[0];

        R_vec = face_centers[n_face];
        R = R_vec.norm();

        l_vec[0] = u_in[1];
        l_vec[1] = u_in[2];
        l_vec[2] = u_in[3];

        vel=cross_product(edge_center, l_vec);
        vel /= (-u_in[0]);
        //vel/=edge_center.norm()*edge_center.norm();



        ndv = dot_product(edge_normals[n_face][n_edge], vel);

        
        //nxR = cross_product(edge_normals[n_face][n_edge], edge_center);

        nxR = cross_product(edge_normals[n_face][n_edge], edge_center);


        /*if(n_face==0){
            std::cout<<dot_product(vel, edge_normals[n_face][n_edge])<<std::endl;
            l_vec.print();
            vel.print();
            edge_normals[n_face][n_edge].print();

        }*/

        res[0] = u_in[0] * dot_product(vel, edge_normals[n_face][n_edge]);
        res[1] = u_in[1] * ndv - nxR[0] * PI;
        res[2] = u_in[2] * ndv - nxR[1] * PI;
        res[3] = u_in[3] * ndv - nxR[2] * PI;

        return res;
    }


    StateVec source(StateVec& u, int n_face){
        StateVec res;
        vector3d<double> fc_normed, l_vec, vel, omxv, rxomxv, omxv1;

        for (size_t i = 0; i < DIM; i++)
            res[i]=0;


        l_vec[0] = u[1];
        l_vec[1] = u[2];
        l_vec[2] = u[3];

        fc_normed = face_centers[n_face] / face_centers[n_face].norm();
        vel = cross_product(fc_normed, l_vec);
        vel /= (-u[0]);

        double theta = std::acos(fc_normed[2]);
        double phi = std::atan2(fc_normed[1],fc_normed[0]);


            if(non_inertial_rf_on){
            omxv=cross_product(omega0, vel);
        
            //if(omxv.norm()>1e-19){
                omxv1 = omxv - fc_normed * dot_product(omxv, fc_normed);
                //std::cout << dot_product(omxv, fc_normed) <<"\n";
            //}
            //else{
                //omxv1=omxv;
            //}


            rxomxv = cross_product(fc_normed, omxv1);
            res[1] = -2*u[0]*rxomxv[0];
            res[2] = -2*u[0]*rxomxv[1];
            res[3] = -2*u[0]*rxomxv[2];



    
            const double theta_c=M_PI/4;
            const double phi_c=M_PI/4;


            
            //const double R0=2./11;
            const double R0=2./(11*2);

            //const double masspwr=8e-3; //v1
            //const double masspwr=2e-2;

            //const double masspwr=8e-2;
            //const double masspwr=-8e-2;


            //const double masspwr=0.15;
            //const double masspwr=0.15/75;
            //const double masspwr=1.;
            //const double masspwr=3.;

            //const double masspwr=8e-4;
            //const double masspwr=-8e-4;

            //const double masspwr=8e-3;
            //const double masspwr=-8e-2;
            //const double masspwr=8e-2;


            //const double masspwr=8e-4;


            //const double masspwr=1e-7;

            const double masspwr=0;

            //const double masspwr=-2e-2; //for bigger r
            //const double masspwr=2e-2;
            //const double masspwr=6e-1;
            //const double masspwr=2.;
            

            //Gaussian heat source for test
            double lon=-theta+M_PI/2;
            double r=2*std::acos(std::sin(theta_c)*std::sin(lon)+std::cos(lon)*std::cos(theta_c)*std::cos(phi-phi_c));
            
           //double t_stop=214;
            //double t_stop=100;
            
            // double t_stop=30;


            // if(r<R0 && t<= t_stop){  //t<=150){
            //     res[0]+=  masspwr * std::exp(-r*r/(2*R0*R0));
            //     }

                        // double t_stop=30;


            double t_med=60;
            double d_disp=30;
            double adj_cf=1;
            if(r<R0 && t<= t_med+2*d_disp){  //t<=150){
                res[0]+= adj_cf* masspwr * std::exp(-r*r/(2*R0*R0))*std::exp(-std::pow(t-t_med,2)/(2*d_disp*d_disp));
                }
        }


    return res;

    };



    std::array<double, 2> char_vel(StateVec u_L, StateVec u_R, int n_face, int n_edge)
    {
        // returns vector {S_L, S_R}
        std::array<double, 2> res;
        double a_L, a_R, S_L, S_R, p_L, p_R;
        vector3d<double> vel_r, vec_r, vel_l,  vec_l, edge_center_l, edge_center_r;

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center_r = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center_r/=edge_center_r.norm();
        edge_center_l = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center_l/=edge_center_l.norm();

        vec_l[0] = u_L[1];
        vec_l[1] = u_L[2];
        vec_l[2] = u_L[3];

        vec_r[0] = u_R[1];
        vec_r[1] = u_R[2];
        vec_r[2] = u_R[3];

        vel_l=cross_product(edge_center_l, vec_l);
        vel_l /= (-u_L[0])*edge_center_l.norm();

        vel_r=cross_product(edge_center_r, vec_r);
        vel_r /= (-u_R[0])*edge_center_r.norm();



        p_L = a * a * u_L[0];
        p_R = a * a * u_R[0];

        a_L = a;
        a_R = a;


        S_L = std::min(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) - std::max(a_L, a_R);
        S_R = std::max(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) + std::max(a_L, a_R);

        res[0] = S_L;
        res[1] = S_R;

        return res;
    }

    double extra_dt_constr()
    {
        double dt_new=1e20;

        return dt_new;
    }

    // StateVec limiter(StateVec& u_r, int n_face, int n_edge)
    // { // classical Superbee limiter for irregular grids
    //     // CFL independent
    //     double etha_minus, etha_plus;
    //     vector3d<double> R_vec, l_vec, vel, vel1, vel2, edge_center;
    //     double R, c, nu_plus;
    //     StateVec res;

    //     int n_edge_1 = n_edge + 1;
    //     if ((n_edge_1) == faces[n_face].size())
    //     {
    //         n_edge_1 = 0;
    //     }

    //     edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
    //     edge_center/=edge_center.norm();



    //     l_vec[0] = U[n_face][1];
    //     l_vec[1] = U[n_face][2];
    //     l_vec[2] = U[n_face][3];

    //     vel=cross_product(edge_center, l_vec);
    //     vel /= -U[n_face][0];

    //     c = a;

    //     nu_plus = (c + dot_product(vel, edge_normals[n_face][n_edge])) * dt *
    //               (distance(vertices[faces[n_face][n_edge]],vertices[faces[n_face][n_edge_1]]) / surface_area[n_face]);

    //     etha_plus = H_plus[n_face][n_edge] / BM_dist[n_face][n_edge];
    //     etha_plus = H_minus[n_face][n_edge] / BM_dist[n_face][n_edge];

    //     for (size_t i = 0; i < DIM; i++)
    //     {
    //         res[i] = std::max(0.,
    //                           std::max(std::min(1., etha_minus * u_r[i] * 2 / (2 * faces[n_face].size() * nu_plus)),
    //                                    std::min(u_r[i], etha_plus)));
    //         /*res[i] = std::max(0.,
    //         std::max(std::min(1., etha_minus * u_r[i]),
    //         std::min(u_r[i], etha_plus)));*/

    //         if (std::isnan(u_r[i]))
    //         {
    //             res[i] = 0;
    //         }
    //     }

    //     return res;
    // };

        StateVec limiter(StateVec& u_r, int n_face, int n_edge)
    { // here U[4] is also pressure

        StateVec res;

        double a = 4, b = 2, c = 0.1, d = 10, e = 3, f = 6; // switch function parameters
        auto h = [a, b, c, d, e, f](double r)
        {
            double res = 0;
            if (r < 1 && r > 0)
                res = (1 - std::tanh(a * std::pow(r, b) * std::pow(1 - r, c)));
            if (r >= 1)
                res = std::pow(std::tanh(d * std::pow(r - 1, e)), f);

            return res;
        };

        StateVec to = limiter_third_order(u_r, n_face, n_edge);
        StateVec sb = limiter_superbee(u_r, n_face, n_edge);

        for (size_t i = 0; i < DIM; i++)
        {
            res[i] = ((1 - h(u_r[i])) * to[i] + h(u_r[i]) * sb[i]);
            // res[i] = ((1 - h(u_r[0])) * to[0] + h(u_r[0]) * sb[0]);
            //  res[i] = 0;
            if (std::isnan(u_r[i]))
            {
                res[i] = 0;
            }
            // res[i]=1;
        }

        return limiter_third_order(u_r, n_face, n_edge);
        // return limiter_superbee(u_r, n_face, n_edge);
        // return res;
    }

    StateVec limiter_third_order(StateVec u_r, int n_face, int n_edge)
    { // here U[4] is also pressure
        StateVec supb = limiter_superbee(u_r, n_face, n_edge);
        vector3d<double> R_vec, l_vec, vel, edge_center;
        double R, c, nu_plus;
        StateVec res;

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
        vel /= (-U[n_face][0] - rho_an[n_face]) * edge_center.norm();
        // double p = pressure(U[n_face], vel, edge_center);

        double gam_0 = make_gam(u_r, edge_center);

        c = a;

        nu_plus = (c + dot_product(vel, edge_normals[n_face][n_edge])) * dt *
                  (distance(vertices[faces[n_face][n_edge]], vertices[faces[n_face][n_edge_1]]) / surface_area[n_face]);

        for (size_t i = 0; i < DIM; i++)
        {

            res[i] = std::max(0., std::min(supb[i], 1 + (1 + nu_plus) / 3 * (u_r[i] - 1)));

            if (std::isnan(u_r[i]))
            {
                res[i] = 0;
            }
        }

        return res;
    }

    StateVec limiter_superbee(StateVec u_r, int n_face, int n_edge)
    { // here U[4] is also pressure
        // classical Superbee limiter for irregular grids
        // CFL independent
        double etha_minus, etha_plus;
        vector3d<double> R_vec, l_vec, vel, edge_center;
        double R, c, nu_plus;
        StateVec res;

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
        vel /= (-U[n_face][0] - rho_an[n_face]) * edge_center.norm();
        // double p = pressure(U[n_face], vel, edge_center);

        c = a;

        nu_plus = (c + dot_product(vel, edge_normals[n_face][n_edge])) * dt *
                  (distance(vertices[faces[n_face][n_edge]], vertices[faces[n_face][n_edge_1]]) / surface_area[n_face]);

        etha_plus = H_plus[n_face][n_edge] / BM_dist[n_face][n_edge];
        etha_minus = H_minus[n_face][n_edge] / BM_dist[n_face][n_edge];

        for (size_t i = 0; i < DIM; i++)
        {

            res[i] = std::max(0.,
                              std::max(std::min(1., etha_minus * u_r[i] / (2 * faces[n_face].size() * nu_plus)),
                                       std::min(u_r[i], etha_plus)));
            if (std::isnan(u_r[i]))
            {
                res[i] = 0;
            }
        }

        return res;
    };

    void set_analytical_solution()// analytical solution to be preserved                               
    {                             // if no AS is required, thish should set rho_an and p_an to 0
        vector3d<double> vec_l, vel,r;
        for (size_t i = 0; i < faces.size(); i++)
        {
            vec_l[0] = U[i][1];
            vec_l[1] = U[i][2];
            vec_l[2] = U[i][3];


            vel = cross_product(face_centers[i]/face_centers[i].norm(), vec_l);
            vel /= -U[i][0];

            //p_an[i] = pressure(U[i], vel, face_centers[i]);
            //rho_an[i] = U[i][0];   //will try to conserve current profile

            rho_an[i] = 0;   //no profile to be conserved
            p_an[i] = 0;
        }
    }

    double make_gam(StateVec &u, vector3d<double> &r){ //crutches
        return 0;
    }

};