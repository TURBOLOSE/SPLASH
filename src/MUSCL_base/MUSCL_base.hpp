#pragma once

#include "../geometry/MUSCL_geometry.hpp"
#include "../json.hpp"
#include <omp.h>
#include <array>
#include <limits>
#include <stdexcept>
#include "../numerical_safety.hpp"
#include "../physics/pressure_conversion.hpp"

//constexpr size_t DIM = 8;
constexpr size_t DIM = 5;
//constexpr size_t DIM = 4;
using StateVec = std::array<double, DIM>;


class MUSCL_base : public MUSCL_base_geometry
{
protected:
    std::vector<StateVec> U, U_temp, U_temp_1, source_plus;
    std::vector<double> rho_an, p_an;
    std::vector<int> reconstruction_failed;
    std::vector<std::vector<StateVec>> flux_var_plus, flux_var_minus, U_plus, U_minus;
    // flux_var^plus_ij flux_var^minus_ij, U_ij (short), U_ji(short)
    double dt = 0., gam, M, N, h0, t, max_vel, rho_full, E_full, c_s, CFL,a, min_dt=1e-10;
    // pressure_fc() is called while bootstrapping the initial floors.
    double density_floor = 0., Y_floor = 0., pressure_floor = 0.;
    bool enforce_tangent_angular_momentum = false;
    bool project_restart_angular_momentum = false;
    double angular_momentum_tolerance = 1e-5;
    std::array<double, 3> initial_projection_L{}, projection_L{}, projection_first{}, projection_last{};
    size_t diagnostic_stage = 0;
    int implicit_iternum;
    bool var_gamma, implicit_solve_on, non_inertial_rf_on, mag_field_on, centrifugal_force_on, nuclear_burning_on;
    size_t steps, threads;
    double omega_ns;
    bool stop_check = false;
    vector3d<double> omega0;


public:
    MUSCL_base(SurfaceMesh mesh, std::vector<StateVec> U_in, double gam, double omega_ns_i, size_t threads_i) : MUSCL_base_geometry(mesh), U(U_in), gam(gam), omega_ns(omega_ns_i), threads(threads_i)
    { // U_in should be n_faces * DIM(=4)

        double h0_temp;

        if (U_in.size() != mesh.n_faces())
        {
            throw std::invalid_argument("U_in.size() must equal mesh.n_faces()");
        }
        size_t nf=this->n_faces();
        // memory allocation
        flux_var_plus.resize(nf);
        flux_var_minus.resize(nf);
        U_plus.resize(nf);
        U_minus.resize(nf);
        U_temp.resize(nf);
        U_temp_1.resize(nf);
        source_plus.resize(nf);
        rho_an.resize(nf);
        p_an.resize(nf);
        reconstruction_failed.resize(nf);

        std::fill(rho_an.begin(), rho_an.end(), 0);
        std::fill(p_an.begin(), p_an.end(), 0);

        for (size_t i = 0; i < nf; i++)
        {

            flux_var_plus[i].resize(faces[i].size());
            flux_var_minus[i].resize(faces[i].size());
            U_plus[i].resize(faces[i].size());
            U_minus[i].resize(faces[i].size());
        }

        std::ifstream ifs("input/parameters.json");
        nlohmann::json parameters = nlohmann::json::parse(ifs);

        CFL = parameters["CFL"];
        var_gamma = parameters["var_gamma"];
        implicit_iternum = parameters["implicit_iternum"];
        implicit_solve_on = parameters["implicit_solve_on"];
        non_inertial_rf_on = parameters["non_inertial_rf_on"];
        mag_field_on = parameters["magnetic_field_on"];
        centrifugal_force_on = parameters["centrifugal_force_on"];
        a=parameters["a_isothermal"];
        omega_ns = parameters["omega_ns"];
        nuclear_burning_on = parameters["nuclear_burning_on"];
        if (DIM != 4 && (!(gam > 1.) || !(gam < 2.) || !std::isfinite(gam)))
            throw std::invalid_argument("Adiabatic gamma must be finite and between one and two");
        enforce_tangent_angular_momentum = parameters.value("enforce_tangent_angular_momentum", false);
        project_restart_angular_momentum = parameters.value("project_restart_angular_momentum", false);
        angular_momentum_tolerance = parameters.value("angular_momentum_tolerance", 1e-5);
        if (!(angular_momentum_tolerance > 0.) || !std::isfinite(angular_momentum_tolerance))
            throw std::invalid_argument("angular_momentum_tolerance must be finite and positive");
        if (project_restart_angular_momentum && !enforce_tangent_angular_momentum)
            throw std::invalid_argument("Restart projection requires tangent angular-momentum enforcement");
        const double initial_radial_fraction = angular_momentum_radial_fraction();
        if (!std::isfinite(initial_radial_fraction))
            throw std::invalid_argument("Nonfinite initial angular momentum");
        if (enforce_tangent_angular_momentum &&
            initial_radial_fraction > angular_momentum_tolerance && !project_restart_angular_momentum)
            throw std::invalid_argument("Initial L is not tangent: use a clean initial state or explicitly set project_restart_angular_momentum=true");
        if (enforce_tangent_angular_momentum) {
            for (size_t i = 0; i < nf; ++i) {
                auto change = splash::project_angular_momentum(U[i], face_centers[i]);
                for (size_t k = 0; k < 3; ++k)
                    initial_projection_L[k] += surface_area[i]*change[k];
            }
        } else if (initial_radial_fraction > angular_momentum_tolerance) {
            std::cerr << "Initial radial-L RMS fraction = " << initial_radial_fraction
                      << "; tangent enforcement is disabled (legacy transport).\n";
        }




        if(mag_field_on && DIM!=8){
            std::cout<<"magnetic field on but DIM is not 8, check MUSCL_base.hpp \n";
            stop_check=true;
            exit(1);
        }

        if(nuclear_burning_on && DIM!=6){
            std::cout<<"nuclear burning on but DIM is not 6, check MUSCL_base.hpp \n";
            stop_check=true;
            exit(1);
        }
        
        omega0[0] = 0;
        omega0[1] = 0;
        omega0[2] = omega_ns;

        N = 1;   // very small face number to be changed
        h0 = 10; // very big length of an edge to be changed
        for (size_t n_face = 0; n_face < nf; n_face++)
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

    void set_min_values()
    {

        double min_p = 1e100, min_rho = 1e100, p;
        size_t nf=this->n_faces();
        for (size_t i = 0; i < nf; i++)
        {
            if (!(U[i][0] > 0.) || !std::isfinite(U[i][0]))
                throw std::invalid_argument("Initial surface density must be finite and positive");
            if (U[i][0] < min_rho)
                min_rho = U[i][0];

            p = pressure_fc(U[i], i);
            if (!(p > 0.) || !std::isfinite(p))
                throw std::invalid_argument("Initial recovered pressure must be finite and positive");
            if (p < min_p)
                min_p = p;
        }

        density_floor = min_rho * 1e-9;
        pressure_floor = min_p * 1e-9;
        Y_floor=0;
    };



    void do_step(double dt0)
    { // RK2 or implicit trapezioidal
        if (!(dt0 > 0.) || !std::isfinite(dt0))
            throw std::invalid_argument("Timestep must be finite and positive");

        size_t nf=this->n_faces();
        dt = dt0;
        diagnostic_stage = 0;
        begin_source_diagnostics();
        U_temp = U;

        find_v_max();

        // h0 = typical length of an edg
        if (dt > h0 * CFL / max_vel)
        {
            dt = h0 * CFL / max_vel;
        }
        //std::cout<<"dt="<<dt<<" h0="<<h0 <<" max_vel="<<max_vel<<"\n";

        double extra_dt = extra_dt_constr();


        if (dt > extra_dt)
            dt = extra_dt;

        if(dt<min_dt)
        {
            std::cout<<"dt is too small, simulation stopped \n";
            stop_check=true;
            exit(1);
        }
        

        if (implicit_solve_on)
        {
            //first iteration is different
            //initial U is euler explicit
            
            
            /*
            find_U_edges();
            find_flux_var();
            res2d(dt); //  U = dt*phi(U)
            
            for (size_t i = 0; i < this->n_faces(); i++)
            {

                if (U[i][0] < density_floor)
                    U[i][0] = density_floor; // density floor
            }

            U_temp_1 = U;
            find_U_edges();
            find_flux_var();
            res2d(dt); //  U = dt*phi(dt*phi(U))
            
            //now U = dt*phi(dt*phi(U)); U_temp=U (initial); U_temp_1=dt*phi(U)


            for (size_t i = 0; i < this->n_faces(); i++)
                {

                    for (size_t k = 0; k < DIM; k++)
                    {
                        U[i][k] = U_temp[i][k]+U[i][k]/2+U_temp_1[i][k];
                    }
                    if (U[i][0] < density_floor)
                        U[i][0] = density_floor; // density floor
                }*/

            find_U_edges();
            find_flux_var();

            res2d(dt / 2); // res2d makes U = dt/2*phi(U)

            for (size_t i = 0; i < nf; i++)
            {

                for (size_t k = 0; k < DIM; k++)
                {
                    U_temp_1[i][k]=2*U[i][k];
                    U[i][k] += U_temp[i][k];
                }

                if (U[i][0] < density_floor)
                    U[i][0] = density_floor; // density floor

                if(DIM==6){
                    if(U[i][5] < Y_floor)
                        U[i][5] = Y_floor;
                }

            }

            find_U_edges();
            find_flux_var();
            res2d(dt); // U=dt*phi( U+dt/2*phi(U))

            for (size_t i = 0; i < nf; i++)
            {

                for (size_t k = 0; k < DIM; k++)
                {
                    U[i][k] += U_temp[i][k];
                }
                if (U[i][0] < density_floor)
                    U[i][0] = density_floor; // density floor

                if(DIM==6){
                    if(U[i][5] < Y_floor)
                        U[i][5] = Y_floor;
                }
            }

            
            for (size_t iternum = 1; iternum < implicit_iternum; iternum++)
            {
                
                double temp0 = 0;
                find_U_edges();
                find_flux_var();
                res2d(dt); // U = dt*phi(U_prev )

                for (size_t i = 0; i < nf; i++)
                {

                    for (size_t k = 0; k < DIM; k++)
                    {   

                        temp0+=U[i][k];
                        U[i][k] = U[i][k]/2+U_temp_1[i][k]/2+U_temp[i][k]; // U=U+1/2*dt*phi(U)+1/2*dt*phi(dt*phi(U))
                        temp0-=U[i][k];

                        
                    }
                    if (U[i][0] < density_floor)
                        U[i][0] = density_floor; // density floor

                }
                //std::cout<<" Implicit iteration "<<iternum<<" diff= "<<temp0<<std::endl;
            }

        
        }
        else
        {

            find_U_edges();
            find_flux_var();

            res2d(dt / 2); // res2d makes U = dt/2*phi(U)

            for (size_t i = 0; i < nf; i++)
            {

                for (size_t k = 0; k < DIM; k++)
                {
                    U[i][k] += U_temp[i][k];
                }

                if (U[i][0] < density_floor)
                    U[i][0] = density_floor; // density floor
            }

            find_U_edges();
            find_flux_var();
            res2d(dt); // U=dt*phi( U+dt/2*phi(U))

            for (size_t i = 0; i < nf; i++)
            {

                for (size_t k = 0; k < DIM; k++)
                {
                    U[i][k] += U_temp[i][k];
                }
                if (U[i][0] < density_floor)
                    U[i][0] = density_floor; // density floor

                //if(i==3729)
                //    std::cout<<i<<" density ="<<U[i][0]<<"\n";
            }
        }

        // The final accepted RHS is midpoint (or the final trapezoidal iterate).
        const bool trapezoidal = implicit_solve_on && implicit_iternum > 1;
        auto accepted_projection = splash::accepted_source_increment(projection_first, projection_last, dt, trapezoidal);
        for (size_t k = 0; k < 3; ++k) projection_L[k] += accepted_projection[k];
        finish_source_diagnostics(dt, trapezoidal);

        double temp = 0;
        double l1, l2, l3 = 0;
        double temp_E = 0;
        l1 = 0;
        l2 = 0;
        for (size_t i = 0; i < nf; i++)
        {

            // for (size_t k = 0; k < DIM; k++)
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

    double angular_momentum_radial_fraction() const
    {
        double radial = 0., total = 0.;
        for (size_t i = 0; i < U.size(); ++i) {
            for (size_t k = 1; k <= 3; ++k)
                if (!std::isfinite(U[i][k])) return std::numeric_limits<double>::quiet_NaN();
            radial += surface_area[i]*splash::radial_angular_momentum_squared(U[i], face_centers[i]);
            for (size_t k = 1; k <= 3; ++k) total += surface_area[i]*U[i][k]*U[i][k];
        }
        return total > 0. ? std::sqrt(radial/total) : 0.;
    }

    // Offline operator replay: evaluates reconstruction, fluxes and sources at
    // the saved state without accepting a step, changing time, or accumulating
    // source/projection work. Stage caches are refreshed. This is also the
    // regression-test entry point; no do_step() call is needed.
    std::vector<StateVec> evaluate_rhs(double reconstruction_dt)
    {
        if (!(reconstruction_dt > 0.) || !std::isfinite(reconstruction_dt))
            throw std::invalid_argument("Reconstruction timestep must be positive");
        auto original = U;
        const double original_dt = dt;
        const size_t original_stage = diagnostic_stage;
        const auto original_first = projection_first, original_last = projection_last;
        dt = reconstruction_dt;
        diagnostic_stage = 0;
        begin_source_diagnostics();
        try {
            find_U_edges();
            find_flux_var();
            res2d(1.);
            auto rhs = U;
            U = std::move(original);
            dt = original_dt;
            diagnostic_stage = original_stage;
            projection_first = original_first;
            projection_last = original_last;
            return rhs;
        } catch (...) {
            U = std::move(original);
            dt = original_dt;
            diagnostic_stage = original_stage;
            projection_first = original_first;
            projection_last = original_last;
            throw;
        }
    }

    void set_time(double t_new)
    {
        t = t_new;
    }

    bool get_stop_check()
    { // True = stop computations due to error
        return stop_check;
    }

        bool get_m_field_on()
    { // True = stop computations due to error
        return mag_field_on;
    }
    bool get_nuclear_burning_on()
    { // True = stop computations due to error
        return nuclear_burning_on;
    }


protected:
    void res2d(double dt_here) // space step
    // takes data from U matrix
    {

        double round_diff = distance(vertices[faces[0][1]], vertices[faces[0][2]]) / (vertices[faces[0][1]] - vertices[faces[0][2]]).norm();

        double d_pres;
        size_t nf=this->n_faces();
        projection_last.fill(0.);
        for (size_t i = 0; i < nf; i++)
        {
            for (size_t k = 0; k < DIM; k++) // source terms
                U[i][k] = dt_here * source_plus[i][k];

            // int comp=3;
            // double temp_h=dt_here*source_plus[i][comp];

            for (size_t j = 0; j < faces[i].size(); j++)
            {

                int j1 = j + 1;

                if (j == (faces[i].size() - 1))
                    j1 = 0;

                for (size_t k = 0; k < DIM; k++)
                {
                    if (!std::isfinite(flux_var_minus[i][j][k]))
                    {
                        stop_check = true;
                        throw std::runtime_error("Nonfinite numerical flux at face " + std::to_string(i));
                    }

                    U[i][k] -= dt_here * (distance(vertices[faces[i][j]], vertices[faces[i][j1]]) / surface_area[i]) * (flux_var_minus[i][j][k]); // todo:pre-compute distances maybe
                    // U[i][k] -= dt_here * (vertices[faces[i][j]]-vertices[faces[i][j1]]).norm() / surface_area[i] *(flux_var_minus[i][j][k]);

                    // U[i][k] -= round_diff * dt_here * (vertices[faces[i][j]] - vertices[faces[i][j1]]).norm() / surface_area[i] * (flux_var_minus[i][j][k]);
                }
            }
            if (enforce_tangent_angular_momentum) {
                auto change = splash::project_angular_momentum(U[i], face_centers[i]);
                for (size_t k = 0; k < 3; ++k)
                    projection_last[k] += surface_area[i]*change[k]/dt_here;
            }
        }
        if (diagnostic_stage++ == 0) projection_first = projection_last;
    };

    virtual StateVec flux_star(StateVec& ul, StateVec& ur, int n_face, int n_edge) = 0;
    virtual StateVec limiter(StateVec& u_r, int n_face, int n_edge) = 0;
    virtual StateVec source(StateVec& u, int n_face) = 0;
    virtual double make_gam(StateVec &u, vector3d<double> &r) = 0;
    virtual double extra_dt_constr() = 0;
    virtual void begin_source_diagnostics() {}
    virtual void source_diagnostics_ready() {}
    virtual void finish_source_diagnostics(double, bool) {}
    double source_time() const
    {
        return t + (diagnostic_stage == 0 ? 0. : diagnostic_stage == 1 ? .5*dt : dt);
    }
    // virtual void set_analytical_solution();

private:
    void find_M()
    // computes value M = max (d phi/ d U1, - d phi/ d U2 )
    // requires recently updated flux_var_plus, flux_var_minus
    {
        double max, f1, f2;
        max = 0.001;
        size_t nf=this->n_faces();
        for (size_t i = 0; i < nf; i++)
        {
            for (size_t j = 0; j < faces[i].size(); j++)
            {
                for (size_t k = 0; k < DIM; k++)
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
        vector3d<double> R_vec, vel, l_vec, fc_normed;
        size_t nf=this->n_faces();


        double GM = 0.217909;
        double g_eff;
        double k_m = 1.6e-13;     // k/m in V_unit(speed of light)^2/K
        double c_sigma = 4.85e36; // c/sigma_SB in R_unit*t_unit^2*K^4/M_unit
        double beta_switch = 0.5479; // switch point for initial function
        double C_switch = beta_switch / (1 - beta_switch);
        double beta, gam_0,C;
        
        gam_0=gam;

        for (size_t i = 0; i < nf; i++)
        {
            l_vec[0] = U[i][1];
            l_vec[1] = U[i][2];
            l_vec[2] = U[i][3];
            rho = U[i][0];


            if (rho < density_floor)
            {
                rho = density_floor;
                U[i][0] = density_floor;
            }

            fc_normed = face_centers[i] / face_centers[i].norm();
            vel = cross_product(fc_normed, l_vec);
            vel /= rho;

            /*if(non_inertial_rf_on){
                vel += cross_product(omega0, fc_normed);
            }*/

            g_eff = GM - vel.norm() * vel.norm();
            p = pressure_fc(U[i], i);

            if(var_gamma){
            C = 12. / 5 * k_m * rho / (3 * p) * pow(3. / 4 * c_sigma * g_eff * rho, 1. / 4);
                if (C <= C_switch)
                {
                    beta = C / (1 + C);
                }
                else
                {
                    beta = 1 - pow(1 / C, 4);
                }

                double beta_ceil = 1 - 1e-9, beta_floor = 0;

                if (beta > beta_ceil || std::isnan(beta) || std::isinf(beta))
                    beta = beta_ceil;

                beta = beta - (beta / (pow(1 - beta, 1. / 4)) - C) / ((4 - 3 * beta) / (4 * pow(1 - beta, 5.0 / 4.0)));
                beta = beta - (beta / (pow(1 - beta, 1. / 4)) - C) / ((4 - 3 * beta) / (4 * pow(1 - beta, 5.0 / 4.0)));

                if (beta < beta_floor) // beta limitations
                    beta = beta_floor;

                if (beta > beta_ceil || std::isnan(beta) || std::isinf(beta))
                    beta = beta_ceil;
                gam_0 = (10 - 3 * beta) / (8 - 3 * beta);
            }

            if(mag_field_on){
            vector3d<double> B;
                B[0] = U[i][5]; B[1] = U[i][2]; B[2] = U[i][7];

                double B_mag = B.norm();
                double GM = 0.217909; // grav parameter in R_unit^3/t_unit^2
                double g_eff = std::max(GM - vel.norm() * vel.norm(),0.);
                double H = (2*gam-1)/(gam-1)*p/(rho*g_eff);


                c_s=std::max(std::sqrt(gam_0 * p / rho), std::sqrt(B_mag * B_mag / (4*M_PI*rho*H)));

            }else if(DIM==4){
                c_s=a;

            }else
                {
                c_s = std::sqrt(gam_0 * p / rho);

            }

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
        //std::cout<<max<<"\n";

        // std::cout<< max_vel<< " " << max_Mach<<"\n";
    }

    void find_flux_var()
    {
        size_t nf=this->n_faces();
        omp_set_dynamic(0);           // Explicitly disable dynamic teams
        omp_set_num_threads(threads); // Use 8 threads for all consecutive parallel regions
#pragma omp parallel for
        for (int i = 0; i < nf; i++)
        {
            for (int j = 0; j < faces[i].size(); j++)
            {
                flux_var_minus[i][j] = flux_star(U_plus[i][j], U_minus[i][j], i, j);
            }
            source_plus[i] = source(U[i], i);
        }
        source_diagnostics_ready();
    }

    void find_U_edges() // finding U_ij and U_ji
    {

        // std::vector<double> pp, pm, lim;
        size_t nf=this->n_faces();
        std::fill(reconstruction_failed.begin(), reconstruction_failed.end(), 0);
        for (size_t i = 0; i < nf; i++) // tag1
        {
            if (!(U[i][0] > 0.) || !std::all_of(U[i].begin(), U[i].end(), [](double x) {return std::isfinite(x);}))
                throw std::runtime_error("Invalid conservative stage state at face " + std::to_string(i));
            if (DIM == 5) {
                vector3d<double> ell;
                for (size_t k = 0; k < 3; ++k) ell[k] = U[i][k+1];
                auto rr = face_centers[i]/face_centers[i].norm();
                auto vv = cross_product(rr, ell)/U[i][0];
                if (!(U[i][4] > .5*U[i][0]*vv.norm()*vv.norm()))
                    throw std::runtime_error("Nonpositive internal energy at face " + std::to_string(i));
            }

            U[i][0] -= rho_an[i];

            if(DIM!=4){
                U[i][4] = pressure_fc(U[i], i);
                U[i][4] -= p_an[i];
            }
        }

        omp_set_dynamic(0);           // Explicitly disable dynamic teams
        omp_set_num_threads(threads); // Use threads for all consecutive parallel regions

#pragma omp parallel for
        for (size_t i = 0; i < nf; ++i)
        {
            for (size_t j = 0; j < faces[i].size(); ++j)
            {

                int neighboor_num = neighbors_edge[i][j];
                int j0 = std::find(neighbors_edge[neighboor_num].begin(), neighbors_edge[neighboor_num].end(), i) - neighbors_edge[neighboor_num].begin();

                StateVec pp = p_plus(i, j);
                StateVec pm = p_minus(i, j);
                StateVec r;
                //r.resize(DIM);

                for (size_t k = 0; k < DIM; k++)
                {
                    r[k] = pm[k] / pp[k];

                    // if(abs(pm[k])<1e-10 || abs(pp[k])<1e-10 )// crutch fix for limiter
                    //     r[k]=0;
                }

                StateVec lim = limiter(r, i, j);
                for (size_t k = 0; k < DIM; k++)
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

                        try {
                            U_plus[i][j][k] = E_edge(U_plus[i][j], i, j); // p -> E (tag1)
                        } catch (const std::exception&) {
                            reconstruction_failed[i] = 1;
                            U_plus[i][j][k] = 0.;
                        }
                        // std::cout<<U_plus[i][j][k]<<"\n";
                    }
                    else
                    {
                        U_plus[i][j][k] = U[i][k] + pp[k] * lim[k] * BM_dist[i][j];
                    }

                    // U_plus[i][j][k] = U[i][k] + pp[k] * (kappa+1)/2. * BM_dist[i][j]+pm[k] * (1-kappa)/2. * BM_dist[i][j];
                }
                if (enforce_tangent_angular_momentum) {
                    const auto edge_r = vertices[faces[i][j]] + vertices[faces[i][(j+1)%faces[i].size()]];
                    splash::project_angular_momentum(U_plus[i][j], edge_r);
                }
                U_minus[neighboor_num][j0] = U_plus[i][j];
            }
        }
        for (size_t i = 0; i < nf; ++i)
            if (reconstruction_failed[i])
                throw std::runtime_error("Nonphysical edge reconstruction or failed pressure inversion at face " + std::to_string(i));
    };

    StateVec U_H_plus(int n_face, int face_edge)
    {

        StateVec res;

        for (size_t U_element = 0; U_element < DIM; U_element++)
        {
            res[U_element] = betas_plus[n_face][face_edge][0] * U[flux_faces_plus[n_face][face_edge][0]][U_element] +
                             betas_plus[n_face][face_edge][1] * U[flux_faces_plus[n_face][face_edge][1]][U_element];
        }

        return res;
    };

    StateVec U_H_minus(int n_face, int face_edge)
    {
        StateVec res;

        for (size_t U_element = 0; U_element < DIM; U_element++)
        {
            res[U_element] = betas_minus[n_face][face_edge][0] * U[flux_faces_minus[n_face][face_edge][0]][U_element] +
                             betas_minus[n_face][face_edge][1] * U[flux_faces_minus[n_face][face_edge][1]][U_element];
        }

        return res;
    };

    StateVec p_plus(int n_face, int face_edge)
    {
        StateVec res, Up;
        Up = U_H_plus(n_face, face_edge);
        for (size_t U_element = 0; U_element < DIM; U_element++)
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

    StateVec p_minus(int n_face, int face_edge)
    {
        StateVec res1, Um;
        Um = U_H_minus(n_face, face_edge);
        for (size_t U_element = 0; U_element < DIM; U_element++)
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

protected:
    double pressure_fc(StateVec &u, int n_face) // u[4] == energy
    {                   
        
        if(DIM==4){

            return u[0]*a*a;
        }        
        vector3d<double> l_vec, vel, r;
        // double pressure_floor = 1e-16;
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

        //if(non_inertial_rf_on)
        //    vel += cross_product(omega0, r);
        

        // double gam_0 = make_gam(u, r);

        double gam_0 = gam;
        double GM = 0.217909; // grav parameter in R_unit^3/t_unit^2
        double g_eff = std::max(GM - vel.norm() * vel.norm(),0.);
        double H=(2*gam_0-1)/(gam_0-1)*u[4]/(u[0]*g_eff);

        if (var_gamma)
        {
            double GM = 0.217909;
            double g_eff = GM - vel.norm() * vel.norm();
            double c_sigma = 4.85e36; // c/sigma_SB in R_unit*t_unit^2*K^4/M_unit
            double k_m = 1.6e-13;     // k/m in V_unit(speed of light)^2/K
            // new expression for C
            double C = 12. / 5 * k_m * u[0] / (u[4] - u[0] * vel.norm() * vel.norm() / 2) * pow(3. / 4 * c_sigma * g_eff * u[0], 1. / 4);
            double beta_switch = 0.736194670678821; // switch point for initial function
            double C_switch = -1 - 1 / (beta_switch - 1);
            double beta;

            if (C <= C_switch)
            {
                beta = 1 - 1 / (1 + C);
            }
            else
            {
                beta = 1 - pow(2 / C, 4);
            }

            double beta_ceil = 1 - 1e-9, beta_floor = 0;

            if (beta > beta_ceil || std::isnan(beta) || std::isinf(beta))
                beta = beta_ceil;

            beta = beta - (beta / (pow(1 - beta, 1. / 4) * (1 - beta / 2)) - C) / (-(beta * beta + 6 * beta - 8) / (2 * (2 - beta) * (2 - beta) * pow(1 - beta, 5 / 4.)));
            beta = beta - (beta / (pow(1 - beta, 1. / 4) * (1 - beta / 2)) - C) / (-(beta * beta + 6 * beta - 8) / (2 * (2 - beta) * (2 - beta) * pow(1 - beta, 5 / 4.)));

            // double gam3d = 1 / (2 - gam);
            //  gam_0 = gam3d - (gam3d - 4. / 3) / (1 + beta / (3 * (1 - beta) * (gam3d - 1)));
            // gam_0 = 2 - 1 / gam_0; // 2d ver

            if (beta < beta_floor) // beta limitations
                beta = beta_floor;

            if (beta > beta_ceil || std::isnan(beta) || std::isinf(beta))
                beta = beta_ceil;

            gam_0 = (10 - 3 * beta) / (8 - 3 * beta);
        }

        // return (u[4] - u[0] * (vel.norm() * vel.norm() - omega_ns * omega_ns * std::sin(theta) * std::sin(theta)) / 2) * (gam - 1) / gam; // v3 = compressed star + sin

        //if(non_inertial_rf_on){
        //    return std::max(pressure_floor,(u[4] - u[0] * (vel.norm() * vel.norm() - omega_ns * omega_ns * std::sin(theta) * std::sin(theta)) / 2) * (gam - 1)); // v4 = compressed star new gamma
            //return std::max(pressure_floor,(u[4] - u[0] * (vel.norm() * vel.norm() ) / 2) * (gam - 1)  - gam *omega_ns * omega_ns * std::sin(theta) * std::sin(theta)*u[0]/2);
       //}
        //else{

        double res=(u[4] - u[0] * (vel.norm() * vel.norm()) / 2) * (gam_0 - 1);

        if(mag_field_on){
            vector3d<double> B;
            B[0]=u[5]; B[1]=u[6]; B[2]=u[7];

            res -= B.norm()*B.norm()/(8*M_PI*H)*(gam_0 - 1);
        }


        return std::max(pressure_floor, res); 
    }

    double E_edge(StateVec &u, int n_face, int n_edge) // u[4] == pressure
    { // Invert the conservative pressure closure at the reconstructed edge.
        vector3d<double> l_vec, vel, r;

        l_vec[0] = u[1];
        l_vec[1] = u[2];
        l_vec[2] = u[3];

        int n_edge_1 = n_edge + 1;
        if (n_edge == faces[n_face].size() - 1)
            n_edge_1 = 0;

        r = (vertices[faces[n_face][n_edge]] + vertices[faces[n_face][n_edge_1]]);
        r /= r.norm();

        vel = cross_product(r, l_vec);
        vel /= (-u[0]);

        double res = splash::internal_from_pressure(u[0], u[4], vel.norm()*vel.norm(), gam, var_gamma)
                   + u[0]*vel.norm()*vel.norm()/2;
        if(mag_field_on){
        const double g_eff = std::max(0.217909-vel.norm()*vel.norm(),0.);
        const double H=(2*gam-1)/(gam-1)*u[4]/(u[0]*g_eff);
        vector3d<double> B;
        B[0]=u[5]; B[1]=u[6]; B[2]=u[7];
        res+=B.norm()*B.norm()/(8*M_PI*H);
        }

        return res;
    }
};
