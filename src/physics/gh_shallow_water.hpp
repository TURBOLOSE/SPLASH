#pragma once
#include "../MUSCL_base/MUSCL_base.hpp"
#include "../geometry/MUSCL_geometry.hpp"
#include "../physics/adiabatic.hpp"

class gh_shallow_water : public MUSCL_base
{

private:
    std::ofstream outfile, outfile_curl, outfile_p, outfile_omega, outfile_l[3], outfile_mach, outfile_Y;
    const double GM = 0.217909; // grav parameter in R_unit^3/t_unit^2
    const double y =5.4e8;  // g/cm^2 column depth proxy
    //double y =1.4e7;  // g/cm^2 column depth proxy

public:
    gh_shallow_water(SurfaceMesh mesh, std::vector<StateVec> U_in, double gam, size_t threads)
        :MUSCL_base(mesh, U_in, gam,0,threads)
    {

        is_sw=true;
        set_analytical_solution();
        if(DIM>4 && !(DIM==5 && nuclear_burning_on))
        {
            std::cout << "check DIM \n";
            stop_check = true;
        }

        outfile.open(output_path + "h.dat", std::ios::out | std::ios::trunc);
        outfile.close();
        outfile.open(output_path + "h.dat", std::ios::out | std::ios::app);

        outfile_curl.open(output_path + "curl.dat", std::ios::out | std::ios::trunc);
        outfile_curl.close();
        outfile_curl.open(output_path + "curl.dat", std::ios::out | std::ios::app);

        outfile_p.open(output_path + "p.dat", std::ios::out | std::ios::trunc);
        outfile_p.close();
        outfile_p.open(output_path + "p.dat", std::ios::out | std::ios::app);

        outfile_omega.open(output_path + "omega.dat", std::ios::out | std::ios::trunc);
        outfile_omega.close();
        outfile_omega.open(output_path + "omega.dat", std::ios::out | std::ios::app);

        std::string adrs[] = {"Lx.dat", "Ly.dat", "Lz.dat"};

        for (size_t i=0; i < 3; i++)
        {
            outfile_l[i].open(output_path + adrs[i], std::ios::out | std::ios::trunc);
            outfile_l[i].close();
            outfile_l[i].open(output_path + adrs[i], std::ios::out | std::ios::app);
        }

        if (nuclear_burning_on)
        {
            outfile_Y.open(output_path + "Y.dat", std::ios::out | std::ios::trunc);
            outfile_Y.close();
            outfile_Y.open(output_path + "Y.dat", std::ios::out | std::ios::app);
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

    void write_t_Y()
    {
        if (!nuclear_burning_on)
            return;

        outfile_Y << this->time() << "  ";
        size_t nf = this->n_faces();
        for (size_t n_face = 0; n_face < nf; n_face++)
        {
            // isothermal burning state: U = {rho, l1, l2, l3, rhoY}
            outfile_Y << U[n_face][4]/U[n_face][0]   << " ";
        }
        outfile_Y << "\n";
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
            outfile_mach << vel.norm() / std::sqrt(GM*U[n_face][0]) << " ";
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
            double rho=y/(U[n_face][0]*1e7); 
            outfile_p <<    GM/2*U[n_face][0] * U[n_face][0] *rho << " ";
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

    std::vector<double> get_light_curves()
    {
        // rotation around z axis is implied
        std::vector<double> result;

        vector3d<double> obs_vector_0, obs_vector_45, obs_vector_90, obs_vector_180, l_vec, vel;
        obs_vector_0[0] = 9.4 * 3 * 1e19 / 15000;
        obs_vector_0[1] = 0;
        obs_vector_0[2] = 0; // dist = 9400pc (in R_ns)
        obs_vector_45[0] = std::sqrt(9.4 * 3 * 1e19 / 15000);
        obs_vector_45[1] = 0;
        obs_vector_45[2] = std::sqrt(9.4 * 3 * 1e19 / 15000); // dist = 9400pc (in R_ns)
        obs_vector_90[0] = 0;
        obs_vector_90[1] = 0;
        obs_vector_90[2] = 9.4 * 3 * 1e19 / 15000; // dist = 9400pc (in R_ns)
        obs_vector_180[0] = 0;
        obs_vector_180[1] = 0;
        obs_vector_180[2] = -9.4 * 3 * 1e19 / 15000; // dist = 9400pc (in R_ns)

        double flux_tot_0 = 0, flux_tot_45 = 0, flux_tot_90 = 0, flux_tot_180 = 0;
        double phi_fc, theta_fc, d_vec, cos_alpha;
        size_t nf = this->n_faces();

        for (size_t n_face = 0; n_face < nf; n_face++)
        {
            phi_fc = std::atan2(face_centers[n_face][1] / face_centers[n_face].norm(),
                                face_centers[n_face][0] / face_centers[n_face].norm());
            theta_fc = std::acos(face_centers[n_face][2] / face_centers[n_face].norm());

            l_vec[0] = U[n_face][1];
            l_vec[1] = U[n_face][2];
            l_vec[2] = U[n_face][3];

            vel = cross_product(face_centers[n_face] / face_centers[n_face].norm(), l_vec);
            vel /= (-U[n_face][0]);

            // isothermal pressure

            const double PI = GM/2*U[n_face][0] * U[n_face][0];
            double E = PI;

            if (nuclear_burning_on && DIM == 5)
            {
                double rho=y/(U[n_face][0]*10e5); 
                double gam0=2-1/gam;
                const double PI_local = GM/2*U[n_face][0]*U[n_face][0]*rho; 

                const double Q0 = 5.3e21;
                const double kappa0 = 0.03; // opacity in cgs
                const double m_alpha = 6.65e-24; // g
                const double k_b = 1.3807e-16; // erg/K
                const double c_cgs = 3e10; // cm/s
                const double a_ns = 10e5; // cm
                const double g_cgs = 0.217909 * 1e18 / (3.3e-5 * 3.3e-5 * a_ns * a_ns);
                const double eps_alpha = 1.17e-5; // erg
                const double a0 = 7.56e-15; // erg/(cm^3 K^4)

                const double Y = U[n_face][4]/U[n_face][0];

                const double rho5 = rho/1e5;
                const double T8 = m_alpha / k_b * GM*U[n_face][0]*9e20 / 1e8 /3; // in 1e8 K


                const double Q =  a0 * c_cgs * std::pow(T8 * 1e8, 4) / (3 * kappa0 * y * y);

                const double dQ_dt = Q * y * 1e7 * 2.97e-33;
                E = dQ_dt;
            }

            if (phi_fc < M_PI / 2 && phi_fc > -M_PI / 2)
            {
                d_vec = dot_product(obs_vector_0, face_centers[n_face] / face_centers[n_face].norm());
                cos_alpha = std::abs(d_vec) / obs_vector_0.norm();
                flux_tot_0 += E * cos_alpha * surface_area[n_face];
            }

            if (theta_fc < M_PI / 2)
            {
                d_vec = dot_product(obs_vector_90, face_centers[n_face] / face_centers[n_face].norm());
                cos_alpha = std::abs(d_vec) / obs_vector_90.norm();
                flux_tot_90 += E * cos_alpha * surface_area[n_face];
            }

            if (dot_product(obs_vector_45, face_centers[n_face] / face_centers[n_face].norm()) > 0)
            {
                d_vec = dot_product(obs_vector_45, face_centers[n_face] / face_centers[n_face].norm());
                cos_alpha = std::abs(d_vec) / obs_vector_45.norm();
                flux_tot_45 += E * cos_alpha * surface_area[n_face];
            }

            if (theta_fc > M_PI / 2)
            {
                d_vec = dot_product(obs_vector_180, face_centers[n_face] / face_centers[n_face].norm());
                cos_alpha = std::abs(d_vec) / obs_vector_180.norm();
                flux_tot_180 += E * cos_alpha * surface_area[n_face];
            }
        }

        result.push_back(flux_tot_0);
        result.push_back(flux_tot_45);
        result.push_back(flux_tot_90);
        result.push_back(flux_tot_180);
        return result;
    };



public:
    //const double a = 1;


    StateVec flux(StateVec& u_in, int n_face, int n_edge)
    {
        StateVec res;
        double PI, ndv, L, A, R;
        vector3d<double> R_vec, vel, vel1, vel2, l_vec, nxR, edge_center, omxv1;

        if(u_in[0]<density_floor)
            u_in[0]=density_floor;

        int n_edge_1 = n_edge + 1;
        if ((n_edge_1) == faces[n_face].size())
        {
            n_edge_1 = 0;
        }

        edge_center = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;
        edge_center /= edge_center.norm(); //unit sphere


        PI = GM*u_in[0]*u_in[0]/2;

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



        res[0] = u_in[0] * ndv;
        res[1] = u_in[1] * ndv - nxR[0] * PI;
        res[2] = u_in[2] * ndv - nxR[1] * PI;
        res[3] = u_in[3] * ndv - nxR[2] * PI;

    if (nuclear_burning_on)
        res[4] = u_in[4] * ndv; // hY transfer
        //res[4] = u_in[4] * u_in[0] * ndv; // rhoY transfer

        return res;
    }


    StateVec source(StateVec& u, int n_face){

        if(u[0]<density_floor)
            u[0]=density_floor;

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


            rxomxv = cross_product(fc_normed, omxv);
            res[1] = -2*u[0]*rxomxv[0];
            res[2] = -2*u[0]*rxomxv[1];
            res[3] = -2*u[0]*rxomxv[2];            



            // omxv=cross_product(fc_normed, vel);
            // rxomxv=cross_product(fc_normed, omxv);
            // double f0=omega0[2]*std::sqrt(2);
            // double sign_hs=1;
            // // if(theta>M_PI/2)
            // //     sign_hs=-1;

            // double adj_coeff=1;// testing
            
            // res[1] = -2*f0*u[0]*omxv[0]*adj_coeff;
            // res[2] = -2*f0*u[0]*omxv[1]*adj_coeff;
            // res[3] = -2*f0*u[0]*omxv[2]*adj_coeff;  

            //std::cout<<res[1]<<" "<<res[2]<<" "<<res[3]<<"\n";

        }

        // Nuclear burning (ported from adiabatic.hpp).
        // isothermal state with burning: u = {h, l1, l2, l3, rhoY}
        // source returns res[4] =  d(hY)/dt
        if (nuclear_burning_on && DIM == 5)
        {
            double rho=y/(u[0]*1e6); 
            double gam0=2-1/gam;
            double PI_local = GM/2*u[0]*u[0]*rho;

            double Q0 = 5.3e21;
            const double kappa0 = 0.03; // opacity in cgs
            const double m_alpha = 6.65e-24; // g
            const double k_b = 1.3807e-16; // erg/K
            const double c_cgs = 3e10; // cm/s
            const double a_ns = 10e5; // cm
            const double g_cgs = 0.217909 * 1e18 / (3.3e-5 * 3.3e-5 * a_ns * a_ns);
            const double eps_alpha = 1.17e-5; // erg
            const double a0 = 7.56e-15; // erg/(cm^3 K^4)

            double Y = u[4]/u[0];



            const double rho5 = rho/1e5;



            double T8 = m_alpha / k_b * u[0]*1e6 * g_cgs / 1e8 /3; // in 1e8 K
            

            double heat_adj=1;
            // if(T8<1)
            //     heat_adj=0;
            // if(T8>17)
            //     heat_adj=0;

            double Q = heat_adj*(Q0 * rho5 * rho5 * std::pow(Y, 3) / std::pow(T8, 3) * std::exp(-44.027 / T8)
                            - a0 * c_cgs * std::pow(T8 * 1e8, 4) / (3 * kappa0 * y * y));


            
            //erg/(g s) => (cm^2/s^3)/(cm/s^2) =>cm/s
            res[0]+= Q*3.6e-26/GM;
            double sigma_sb=5.6e-5;
            double T_const_heat=0;
            res[0]+=a0 * c_cgs * std::pow(T_const_heat, 4) / (3 * kappa0 * y * y)*3.6e-26/GM; //const heat bonus
            res[4] += -heat_adj*Q0 * rho5 * rho5 * std::pow(Y, 3) / std::pow(T8, 3) * std::exp(-44.027 / T8)* 3 * m_alpha / eps_alpha * 3.3e-5*u[0]+
             Q*3.6e-26/GM*Y; // dhY/dt=dh/dt Y + dY/dt h


            //simple ver
            // double t_nuc=0.3/(3.3e-5);//s to T_unit
            // double t_cool=15/(3.3e-5); 
            // double heat_mult=10;
            // double cool_mult=10;

            // double t_nuc=0.3/(3.3e-5);//s to T_unit
            // double t_cool=15/(3.3e-5); 
            // double heat_mult=1./2;
            // double cool_mult=1./2;


            // if(T8<2){
            //     heat_mult=0;
            //     cool_mult=0;
            // }
            // if(T8>25)
            //     heat_mult=0;

            // res[0] +=heat_mult*Y*u[0]/t_nuc*std::pow(T8,2)-u[0]/t_cool*std::pow(T8,4)*cool_mult;
            // res[4] +=-heat_mult*Y*u[0]/t_nuc*std::pow(T8,2)* 3 * m_alpha / eps_alpha *c_cgs*c_cgs * GM*u[0] + Y*(res[0]);



        }


    return res;

    };



    std::array<double, 2> char_vel(StateVec u_L, StateVec u_R, int n_face, int n_edge)
    {
        // returns vector {S_L, S_R}
        std::array<double, 2> res;
        double a_L, a_R, S_L, S_R;
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


        a_L =std::sqrt(GM*u_L[0]);
        a_R =std::sqrt(GM*u_R[0]);


        S_L = std::min(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) - std::max(a_L, a_R);
        S_R = std::max(dot_product(vel_l, edge_normals[n_face][n_edge]), dot_product(vel_r, edge_normals[n_face][n_edge])) + std::max(a_L, a_R);

        res[0] = S_L;
        res[1] = S_R;

        return res;
    }

    double extra_dt_constr()
    {
        double dt_new=1e20;
        size_t nf = this->n_faces();
        for (size_t i = 0; i < nf; i++)
        {
            if (nuclear_burning_on && DIM == 5)
            {
                double rho=y/(U[i][0]*10e5); 
                double gam0=2-1/gam;
                double PI_local = GM/2*U[i][0]*U[i][0]*rho;

                double Q0 = 5.3e21;
                const double kappa0 = 0.03; // opacity in cgs
                const double m_alpha = 6.65e-24; // g
                const double k_b = 1.3807e-16; // erg/K
                const double c_cgs = 3e10; // cm/s
                const double a_ns = 10e5; // cm
                const double g_cgs = 0.217909 * 1e18 / (3.3e-5 * 3.3e-5 * a_ns * a_ns);
                const double eps_alpha = 1.17e-5; // erg
                const double a0 = 7.56e-15; // erg/(cm^3 K^4)

                double Y = U[i][4]/U[i][0];


                const double rho5 = rho/1e5;

                double T8 = m_alpha / k_b * U[i][0]*1e6 * g_cgs / 1e8 /3; // in 1e8 K
                
                
                double Q = Q0 * rho5 * rho5 * std::pow(Y, 3) / std::pow(T8, 3) * std::exp(-44.027 / T8)
                                - a0 * c_cgs * std::pow(T8 * 1e8, 4) / (3 * kappa0 * y * y);



                double t_temp_1= U[i][0]/(Q*3.6e-26/GM);

                double t_temp_2 = U[i][4]/(-Q0 * rho5 * rho5 * std::pow(Y, 3) / std::pow(T8, 3) * std::exp(-44.027 / T8)* 3 * m_alpha / eps_alpha * 3.3e-5*U[i][0]+
                Q*3.6e-26/GM*Y); // dY/dt = -dQ/dt * m_alpha/eps_alpha * 3.3e-5 + Y/h * dh/dt (isothermal burning, hY transfer)

                if (0.02*std::abs(t_temp_1) < dt_new)
                    dt_new = 0.02*std::abs(t_temp_1);
                if (0.02*std::abs(t_temp_2) < dt_new)
                    dt_new = 0.02*std::abs(t_temp_2);
            }
        }
        
        //if(t>900)
        //dt_new=1e20;

        return dt_new;
    }



        StateVec limiter(StateVec& u_r, int n_face, int n_edge)
    { 
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


            return limiter_venkatakrishnan(u_r, n_face, n_edge);
            //return limiter_third_order(u_r, n_face, n_edge);
             //return limiter_superbee(u_r, n_face, n_edge);
             //return res;
    }

        StateVec limiter_venkatakrishnan(StateVec u_r, int n_face, int n_edge)
        {
            StateVec sb = limiter_superbee(u_r, n_face, n_edge);
            vector3d<double> l_vec, vel, edge_center;
            double c, nu_plus;
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

            c = std::sqrt(GM * u_r[0]);

            nu_plus = (c + dot_product(vel, edge_normals[n_face][n_edge])) * dt *
                      (distance(vertices[faces[n_face][n_edge]], vertices[faces[n_face][n_edge_1]]) / surface_area[n_face]);

            const double eps = 1e-12 * (1.0 + nu_plus * nu_plus);

            for (size_t i = 0; i < DIM; i++)
            {
                double r = std::max(0.0, u_r[i]);
                double venkat = (r * r + 2.0 * r + 1.0 + eps) / (r * r + r + 2.0 + eps);
                res[i] = std::max(0.0, std::min(sb[i], venkat));

                if (std::isnan(u_r[i]))
                {
                    res[i] = 0;
                }
            }

            return res;
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

        c = std::sqrt(GM*u_r[0]);


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
    { 
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

        c = std::sqrt(GM*u_r[0]);

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
