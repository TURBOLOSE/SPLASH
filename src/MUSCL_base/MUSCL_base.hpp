#pragma once

#include "../geometry/MUSCL_geometry.hpp"
#include "../json.hpp"
#include <omp.h>

class MUSCL_base : public MUSCL_base_geometry
{
protected:
    std::vector<std::vector<double>> U, U_temp, source_plus;
    std::vector<double> rho_an, p_an;
    std::vector<std::vector<std::vector<double>>> flux_var_plus, flux_var_minus, U_plus, U_minus;
    // flux_var^plus_ij flux_var^minus_ij, U_ij (short), U_ji(short)
    double dt, gam, M, N, h0, t, max_vel, rho_full, E_full, c_s, density_floor, pressure_floor, CFL;
    int dim;
    bool var_gamma;
    size_t steps, threads;
    double omega_ns;
    bool stop_check = false;

public:
    MUSCL_base(SurfaceMesh mesh, std::vector<std::vector<double>> U_in, int dim, double gam, double omega_ns_i, size_t threads_i) : MUSCL_base_geometry(mesh), U(U_in), gam(gam), dim(dim), omega_ns(omega_ns_i), threads(threads_i)
    { // U_in should be n_faces * dim(=4)

        double h0_temp;

        if (U_in.size() != mesh.n_faces())
        {
            std::cout << "U_in is wrong size, U_in.size() should be equal to mesh.n_faces()" << std::endl;
        }

        // memory allocation
        flux_var_plus.resize(this->n_faces());
        flux_var_minus.resize(this->n_faces());
        U_plus.resize(this->n_faces());
        U_minus.resize(this->n_faces());
        U_temp.resize(this->n_faces());
        source_plus.resize(this->n_faces());
        rho_an.resize(this->n_faces());
        p_an.resize(this->n_faces());

        std::fill(rho_an.begin(), rho_an.end(), 0);
        std::fill(p_an.begin(), p_an.end(), 0);


        
        for (size_t i = 0; i < this->n_faces(); i++)
        {

            flux_var_plus[i].resize(faces[i].size());
            flux_var_minus[i].resize(faces[i].size());
            U_plus[i].resize(faces[i].size());
            U_minus[i].resize(faces[i].size());
            U_temp[i].resize(dim);
            source_plus[i].resize(dim);

            for (size_t j = 0; j < faces[i].size(); j++)
            {
                flux_var_plus[i][j].resize(dim);
                flux_var_minus[i][j].resize(dim);
                U_plus[i][j].resize(dim);
                U_minus[i][j].resize(dim);
            }


        }

        std::ifstream ifs("input/parameters.json");
        nlohmann::json parameters = nlohmann::json::parse(ifs);

        CFL = parameters["CFL"];
        var_gamma = parameters["var_gamma"];



        N = 1;
        h0 = 10;
        for (size_t n_face = 0; n_face < this->n_faces(); n_face++)
        {
            if (faces[n_face].size() > N)
            {
                N = faces[n_face].size();
            }

            for (size_t face_element = 0; face_element < faces[n_face].size(); face_element++)
            {

                if (face_element < faces[n_face].size() - 1)
                {
                    h0_temp = distance(vertices[faces[n_face][face_element]], vertices[faces[n_face][face_element + 1]]);
                    // /surface_area[n_face];
                }
                else
                {
                    h0_temp = distance(vertices[faces[n_face][face_element]], vertices[faces[n_face][0]]);
                    // /surface_area[n_face];
                }

                if (h0_temp < h0)
                {
                    h0 = h0_temp;
                }
            }
        }

        t = 0;
        steps = 0;
       
    };

    void set_min_values(){

        double min_p=1e100, min_rho=1e100,p;
        
        for (size_t i = 0; i < this->n_faces(); i++)
        {
            if(U[i][0]<min_rho)
                min_rho=U[i][0];

            p=pressure_fc(U[i], i);
            if(p<min_p)
                min_p=p;

        }

        density_floor=min_rho*1e-9;
        pressure_floor=min_p*1e-9;

    };

    void do_step(double dt0)
    { // RK2

        dt = dt0;
        U_temp = U;

        find_v_max();

        // h0 = typical length of an edg
        if (dt > h0 * CFL / max_vel)
        {
            dt = h0 * CFL / max_vel;
        }

        double extra_dt = extra_dt_constr();

        if (dt > extra_dt)
            dt = extra_dt;

        //std::cout<<extra_dt<<"\n";


        // if(steps==0)
        // dt=1e-2;

        find_U_edges();
        find_flux_var();

        res2d(dt / 2); // res2d makes U = dt/2*phi(U)
        // res2d(dt);

        for (size_t i = 0; i < this->n_faces(); i++)
        {

            /*for (size_t k = 0; k < dim; k++)
            {
                U_temp[i][k] += U[i][k]; //U_temp = U+dt*phi(U)
                U[i][k] += U_temp[i][k];  //U = U_temp = U+dt*phi(U)

            }*/

            for (size_t k = 0; k < dim; k++)
            {
                // U_temp[i][k] += U[i][k];
                U[i][k] += U_temp[i][k];
            }

            if (U[i][0] < density_floor)
                U[i][0] = density_floor; // density floor
        }

        find_U_edges();
        find_flux_var();
        res2d(dt); // U=dt*phi( U+dt/2*phi(U))

        // U_res = U+dt*phi(U) + dt/2 * phi(U+dt*phi(U))
        for (size_t i = 0; i < this->n_faces(); i++)
        {

            for (size_t k = 0; k < dim; k++)
            {
                // U[i][k]=U[i][k]/2.+U_temp[i][k];
                U[i][k] += U_temp[i][k];
            }
            if (U[i][0] < density_floor)
                U[i][0] = density_floor; // density floor
        }

        // old RK2
        /*res2d(dt / 2.); // res2d makes U = dt/2*phi(U)
        for (size_t i = 0; i < this->n_faces(); i++)
        {

            for (size_t k = 0; k < dim; k++)
            {
                U[i][k] += U_temp[i][k]; // results in U=U+dt/2*phi(U)
            }
        }

        find_U_edges();
        find_flux_var();
        res2d(dt); // U=dt*phi(U+dt/2*phi(U))*/

        double temp = 0;
        double l1, l2, l3 = 0;
        double temp_E = 0;
        l1 = 0;
        l2 = 0;
        for (size_t i = 0; i < this->n_faces(); i++)
        {

            // for (size_t k = 0; k < dim; k++)
            //{
            //     U[i][k] += U_temp[i][k]; // U=U+dt*phi(U+dt/2*phi(U))
            // }
            temp += U[i][0] * surface_area[i];
            l1 += U[i][1] * surface_area[i];
            l2 += U[i][2] * surface_area[i];
            l3 += U[i][3] * surface_area[i];
            temp_E += U[i][4] * surface_area[i];
            // std::cout<<U[i][0]<<" "<<U[i][1]<<" "<<U[i][2]<<" "<<U[i][3]<<std::endl;
        }

        if (steps == 0)
        {
            rho_full = temp;
            E_full = temp_E;
        }

        std::cout << "t= " << t + dt << " mass_relative_gain=" << temp / rho_full - 1 << " (l/mass)_total_norm= " << sqrt(l1 * l1 + l2 * l2 + l3 * l3)
                  << " l_total = (" << l1 << "," << l2 << "," << l3 << ")" << "\n";

        // std::cout <<t + dt<<" "<< temp / rho_full-1 <<" "<<temp_E / E_full-1 <<"\n";
        // std::cout <<t + dt<<" "<< sqrt(l1 * l1 + l2 * l2 + l3 * l3)<<"\n";

        // std::cout<<std::endl;
        // std::cout <<t<<" "<<1. - temp / rho_full << std::endl;

        t += dt;
        steps++;
    }

    double time()
    {
        return t;
    }

    bool get_stop_check()
    { // True = stop computations due to error
        return stop_check;
    }

protected:
    void res2d(double dt_here) // space step
    // takes data from U matrix
    {

        double round_diff = distance(vertices[faces[0][1]], vertices[faces[0][2]]) / (vertices[faces[0][1]] - vertices[faces[0][2]]).norm();

        double d_pres;
        for (size_t i = 0; i < this->n_faces(); i++)
        {
            for (size_t k = 0; k < dim; k++) // source terms
                U[i][k] = dt_here * source_plus[i][k];

            // int comp=3;
            // double temp_h=dt_here*source_plus[i][comp];

            for (size_t j = 0; j < faces[i].size(); j++)
            {

                int j1 = j + 1;

                if (j == (faces[i].size() - 1))
                    j1 = 0;

                for (size_t k = 0; k < dim; k++)
                {
                    if (std::isnan((flux_var_minus[i][j][k])))
                    {
                        stop_check = true;
                        std::cout << "time: " << t << " face: " << i << " edge: " << j << " NaN in flux detected!" << std::endl;
                        exit(1);
                    }

                    // U[i][k] -= dt_here * (distance(vertices[faces[i][j]],vertices[faces[i][j1]]) / surface_area[i]) *(flux_var_minus[i][j][k]);
                    // U[i][k] -= dt_here * (vertices[faces[i][j]]-vertices[faces[i][j1]]).norm() / surface_area[i] *(flux_var_minus[i][j][k]);
                    U[i][k] -= round_diff * dt_here * (vertices[faces[i][j]] - vertices[faces[i][j1]]).norm() / surface_area[i] * (flux_var_minus[i][j][k]);
                }
            }
        }
    };

    virtual std::vector<double> flux_star(std::vector<double> ul, std::vector<double> ur, int n_face, int n_edge) = 0;
    virtual std::vector<double> limiter(std::vector<double> u_r, int n_face, int n_edge) = 0;
    virtual std::vector<double> source(std::vector<double> u, int n_face) = 0;
    virtual double make_gam(std::vector<double> &u, vector3d<double> &r) = 0;
    virtual double extra_dt_constr() = 0;
    // virtual void set_analytical_solution();

private:
    void find_M()
    // computes value M = max (d phi/ d U1, - d phi/ d U2 )
    // requires recently updated flux_var_plus, flux_var_minus
    {
        double max, f1, f2;
        max = 0.001;
        for (size_t i = 0; i < this->n_faces(); i++)
        {
            for (size_t j = 0; j < faces[i].size(); j++)
            {
                for (size_t k = 0; k < dim; k++)
                {
                    f1 = flux_var_plus[i][j][k] / (U_plus[i][j][k] - U[i][k]);
                    f2 = -flux_var_minus[i][j][k] / (U_minus[i][j][k] - U[i][k]);

                    if (!std::isnan(f1) && !std::isinf(f1) && f1 > max)
                        max = f1;

                    if (!std::isnan(f2) && !std::isinf(f2) && f2 > max)
                        max = f2;
                }
            }
        }

        M = max;
    }
    void find_v_max()
    {
        double max, c, p, rho;
        max = 1e-8;
        double max_Mach = 0;
        vector3d<double> R_vec, vel, l_vec;
        for (size_t i = 0; i < this->n_faces(); i++)
        {
            l_vec[0] = U[i][1];
            l_vec[1] = U[i][2];
            l_vec[2] = U[i][3];
            rho = U[i][0];

            if (rho < density_floor){
                rho = density_floor;
                U[i][0]=density_floor;
            }
            vel = cross_product(face_centers[i] / face_centers[i].norm(), l_vec);
            vel /= rho;
            p = pressure_fc(U[i], i);
            c_s = std::sqrt(gam * p / rho);

            if (vel.norm() > max)
            {
                max = vel.norm();
                // if (this->time() > 62)
                //     std::cout << "vel= " << max << " " << rho << " " << p << "\n";
            }

            if (c_s > max)
            {
                max = c_s;
                // if (this->time() > 62)
                //     std::cout << "c_s= " << max << " " << rho << " " << p << "\n";
            }

            if (vel.norm() / c_s > max_Mach)
                max_Mach = vel.norm() / c_s;
        }

        max_vel = max;

        // std::cout<< max_vel<< " " << max_Mach<<"\n";
    }

    void find_flux_var()
    {

        omp_set_dynamic(0);           // Explicitly disable dynamic teams
        omp_set_num_threads(threads); // Use 8 threads for all consecutive parallel regions
#pragma omp parallel for
        for (int i = 0; i < this->n_faces(); i++)
        {
            for (int j = 0; j < faces[i].size(); j++)
            {
                flux_var_minus[i][j] = flux_star(U_plus[i][j], U_minus[i][j], i, j);
            }
            source_plus[i] = source(U[i], i);
        }
    }

    void find_U_edges() // finding U_ij and U_ji
    {

        // std::vector<double> pp, pm, lim;

        for (size_t i = 0; i < this->n_faces(); i++) // tag1
        {
            U[i][4] = pressure_fc(U[i], i);
            U[i][0] -= rho_an[i];
            U[i][4] -= p_an[i];
        }
        // std::cout<<rho_an[0]<<" "<<p_an[0]<<"\n";

        omp_set_dynamic(0);           // Explicitly disable dynamic teams
        omp_set_num_threads(threads); // Use threads for all consecutive parallel regions

#pragma omp parallel for
        for (size_t i = 0; i < this->n_faces(); ++i)
        {
            for (size_t j = 0; j < faces[i].size(); ++j)
            {

                int neighboor_num = neighbors_edge[i][j];
                int j0 = std::find(neighbors_edge[neighboor_num].begin(), neighbors_edge[neighboor_num].end(), i) - neighbors_edge[neighboor_num].begin();

                std::vector<double> pp = p_plus(i, j);
                std::vector<double> pm = p_minus(i, j);
                std::vector<double> r;
                r.resize(dim);

                for (size_t k = 0; k < dim; k++)
                {
                    r[k] = pm[k] / pp[k];

                    // if(abs(pm[k])<1e-10 || abs(pp[k])<1e-10 )// crutch fix for limiter
                    //     r[k]=0;
                }

                std::vector<double> lim = limiter(r, i, j);
                for (size_t k = 0; k < dim; k++)
                {
                    /*double kappa = (2 * lim[k] - (r[k] + 1)) / (r[k] + 1);
                    if (std::isnan(kappa))
                        kappa = 0;*/

                    if (k == 0)
                    {
                        U_plus[i][j][k] = U[i][k] + rho_an[i] + pp[k] * lim[k] * BM_dist[i][j];
                    }
                    else if (k == 4)
                    {
                        U_plus[i][j][k] = U[i][k] + p_an[i] + pp[k] * lim[k] * BM_dist[i][j];

                        // std::cout<<pm[4]<<" "<<pp[4]<<"  "<<lim[4]<<"\n";

                        U_plus[i][j][k] = E_fc(U_plus[i][j], i, j); // p -> E (tag1)
                        //std::cout<<U_plus[i][j][k]<<"\n";
                    }
                    else
                    {
                        U_plus[i][j][k] = U[i][k] + pp[k] * lim[k] * BM_dist[i][j];
                    }

                    // U_plus[i][j][k] = U[i][k] + pp[k] * (kappa+1)/2. * BM_dist[i][j]+pm[k] * (1-kappa)/2. * BM_dist[i][j];
                    U_minus[neighboor_num][j0][k] = U_plus[i][j][k];
                }
            }
        }
    };

    std::vector<double> U_H_plus(int n_face, int face_edge)
    {

        std::vector<double> res;
        res.resize(dim);

        for (size_t U_element = 0; U_element < dim; U_element++)
        {
            res[U_element] = betas_plus[n_face][face_edge][0] * U[flux_faces_plus[n_face][face_edge][0]][U_element] +
                             betas_plus[n_face][face_edge][1] * U[flux_faces_plus[n_face][face_edge][1]][U_element];
        }

        return res;
    };

    std::vector<double> U_H_minus(int n_face, int face_edge)
    {
        std::vector<double> res;
        res.resize(dim);

        for (size_t U_element = 0; U_element < dim; U_element++)
        {
            res[U_element] = betas_minus[n_face][face_edge][0] * U[flux_faces_minus[n_face][face_edge][0]][U_element] +
                             betas_minus[n_face][face_edge][1] * U[flux_faces_minus[n_face][face_edge][1]][U_element];
        }

        return res;
    };

    std::vector<double> p_plus(int n_face, int face_edge)
    {
        std::vector<double> res, Up;
        res.resize(dim);
        Up = U_H_plus(n_face, face_edge);
        for (size_t U_element = 0; U_element < dim; U_element++)
        {
            if (U_element == 0)
            {
                // res[U_element] = (Up[U_element] - U[n_face][U_element]) / H_plus[n_face][face_edge] - rho_an[n_face];
                res[U_element] = (Up[U_element] - U[n_face][U_element]) / H_plus[n_face][face_edge];
            }
            else if (U_element == 4)
            {
                // res[U_element] = (Up[U_element] - U[n_face][U_element]) / H_plus[n_face][face_edge] - p_an[n_face];
                res[U_element] = (Up[U_element] - U[n_face][U_element]) / H_plus[n_face][face_edge];
            }
            else
            {
                res[U_element] = (Up[U_element] - U[n_face][U_element]) / H_plus[n_face][face_edge];
            }
        }
        return res;
    };

    std::vector<double> p_minus(int n_face, int face_edge)
    {
        std::vector<double> res1, Um;
        res1.resize(dim);
        Um = U_H_minus(n_face, face_edge);
        for (size_t U_element = 0; U_element < dim; U_element++)
        {
            if (U_element == 0)
            {
                // res1[U_element] = (-Um[U_element] + U[n_face][U_element]) / H_minus[n_face][face_edge] - rho_an[n_face];
                res1[U_element] = (-Um[U_element] + U[n_face][U_element]) / H_minus[n_face][face_edge];
            }
            else if (U_element == 4)
            {
                // res1[U_element] = (-Um[U_element] + U[n_face][U_element]) / H_minus[n_face][face_edge] - p_an[n_face];
                res1[U_element] = (-Um[U_element] + U[n_face][U_element]) / H_minus[n_face][face_edge];
            }
            else
            {
                res1[U_element] = (-Um[U_element] + U[n_face][U_element]) / H_minus[n_face][face_edge];
            }
        }
        return res1;
    }

    double pressure_fc(std::vector<double> &u, int n_face) // u[4] == energy
    {                                                      // to do: make state_vector a class and turn this into a method
        vector3d<double> l_vec, vel, r;
        //double pressure_floor = 1e-16;
        l_vec[0] = u[1];
        l_vec[1] = u[2];
        l_vec[2] = u[3];

        /*int n_edge_1 = n_edge + 1;
        if (n_edge == faces[n_face].size() - 1)
            n_edge_1 = 0;

        r = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]) / 2.;*/
        r = face_centers[n_face];
        r /= r.norm();

        double theta = std::acos(r[2]);
        vel = cross_product(r, l_vec);
        vel /= -u[0];

        double gam_0 = make_gam(u, r);

        // return (u[4] - u[0] * (vel.norm() * vel.norm() - omega_ns * omega_ns * std::sin(theta) * std::sin(theta)) / 2) * (gam - 1) / gam; // v3 = compressed star + sin
        return std::max(pressure_floor, (u[4] - u[0] * (vel.norm() * vel.norm()) / 2) * (gam_0 - 1)); // v4
    }

    double E_fc(std::vector<double> &u, int n_face, int n_edge) // u[4] == pressure
    {                                                           // because we reconstruct pressure on edge we needed new formula for beta
        vector3d<double> l_vec, vel, r;

        l_vec[0] = u[1];
        l_vec[1] = u[2];
        l_vec[2] = u[3];

        int n_edge_1 = n_edge + 1;
        if (n_edge == faces[n_face].size() - 1)
            n_edge_1 = 0;

        r = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]);
        r /= r.norm();
        double theta = std::acos(r[2]);

        vel = cross_product(r, l_vec);
        vel /= (-u[0]);

        double gam_0 = gam;

        if (var_gamma)
        {
            double GM = 0.217909;
            double g_eff = GM - vel.norm() * vel.norm();
            double c_sigma = 4.85e36; // c/sigma_SB in R_unit*t_unit^2*K^4/M_unit
            double k_m = 1.6e-13;     // k/m in V_unit(speed of light)^2/K
            // new expression for C
            double C = 12. / 5 * k_m * u[0] / (3 * u[4]) * pow(3. / 4 * c_sigma * g_eff * u[0], 1. / 4);
            double beta_switch = 0.5479; // switch point for initial function
            double C_switch = beta_switch / (1 - beta_switch);
            double beta;

            if (C <= C_switch)
            {
                beta = C / (1 + C);
            }
            else
            {
                beta = 1 - pow(1 / C, 4);
            }
            
            double beta_ceil = 1 - 1e-9, beta_floor = 0;

            if (beta > beta_ceil|| std::isnan(beta)||std::isinf(beta))
            beta = beta_ceil;

            beta = beta - (beta / (pow(1 - beta, 1. / 4)) - C) / ((4 - 3 * beta) / (4 * pow(1 - beta, 5 / 4)));
            beta = beta - (beta / (pow(1 - beta, 1. / 4)) - C) / ((4 - 3 * beta) / (4 * pow(1 - beta, 5 / 4)));
           //double gam3d = 1 / (2 - gam);
           // gam_0 = gam3d - (gam3d - 4. / 3) / (1 + beta / (3 * (1 - beta) * (gam3d - 1)));
            //gam_0 = 2 - 1 / gam_0; // 2d ver

            if (beta < beta_floor ) // beta limitations
            beta = beta_floor;

             if (beta > beta_ceil|| std::isnan(beta)||std::isinf(beta))
            beta = beta_ceil;


            gam_0=(10-3*beta)/(8-3*beta);
            
        }
       

        return 1 / (gam_0 - 1) * u[4] + u[0] * (vel.norm() * vel.norm()) / 2;
    }
};