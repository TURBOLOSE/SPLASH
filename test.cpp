#include <iostream>
#include "HLLE.hpp"
#include "HLLE_p.hpp"
#include "HLLC.hpp"

using namespace pmp;

int main()
{

    //SurfaceMesh mesh = quad_sphere(0);
    //SurfaceMesh mesh = icosphere(4);
    SurfaceMesh mesh = icosphere_hex(4);

    double dt = 0.002;
    size_t maxstep = 2000;
    int dim = 4;
    double gam = 1.4;
    std::ifstream inData("input/input.dat");
    std::vector<std::vector<double>> U_in;
    U_in.resize(mesh.n_faces());
    vector3d<double> vel, L;

    for (size_t i = 0; i < mesh.n_faces(); i++)
    {
        U_in[i].resize(dim);
    }

    double element;
    std::vector<double> temp;

    while (!inData.eof() && inData >> element)
    {
        temp.push_back(element);
    }

    for (size_t i = 0; i < mesh.n_faces(); i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            U_in[i][j] = temp[i * dim + j];
        }
    }

    MUSCL_HLLE test2(mesh, U_in, dim, gam);
    //MUSCL_HLLE_p test2(mesh, U_in, dim, gam);
    // MUSCL_HLLC test2(mesh, U_in, dim, gam);

    // MUSCL_base_geometry test(mesh);

    test2.write_face_centers();
    test2.write_faces();
    test2.write_vertices();
    test2.write_t_rho();
    test2.write_t_curl();

    /*std::vector<double> vec1 = test2.flux(U_in[4], 4, 0);
    std::vector<double> vec2 = test2.flux(U_in[4], 4, 1);
    std::vector<double> vec3 = test2.flux(U_in[4], 4, 2);
    std::vector<double> vec4 = test2.flux(U_in[4], 4, 3);

   std::cout << vec1[0] << " " << vec1[1] << " " << vec1[2] << " " << vec1[3] << std::endl;
    std::cout << vec2[0] << " " << vec2[1] << " " << vec2[2] << " " << vec2[3] << std::endl;
    std::cout << vec3[0] << " " << vec3[1] << " " << vec3[2] << " " << vec3[3] << std::endl;
    std::cout << vec4[0] << " " << vec4[1] << " " << vec4[2] << " " << vec4[3] << std::endl; */

    /*vec1 = test2.flux(U_in[0], 0, 0);
    vec2 = test2.flux(U_in[0], 0, 1);
    vec3 = test2.flux(U_in[0], 0, 2);
    vec4 = test2.flux(U_in[0], 0, 3);

    std::cout << vec1[0] << " " << vec1[1] << " " << vec1[2] << " " << vec1[3] << std::endl;
    std::cout << vec2[0] << " " << vec2[1] << " " << vec2[2] << " " << vec2[3] << std::endl;
    std::cout << vec3[0] << " " << vec3[1] << " " << vec3[2] << " " << vec3[3] << std::endl;
    std::cout << vec4[0] << " " << vec4[1] << " " << vec4[2] << " " << vec4[3] << std::endl;*/



    /*std::vector<double> vec2=test2.flux_star(U_in[2], U_in[4],2,1);
    std::cout<<"HLLE(ul, ur): "<<vec2[0]<<" "<<vec2[1]<<" "<<vec2[2]<<" "<<vec2[3]<<std::endl;
    std::cout<<std::endl;

    vec2=test2.flux_star(U_in[4], U_in[2],4,1);
    std::cout<<"HLLE(ur, ul): "<<vec2[0]<<" "<<vec2[1]<<" "<<vec2[2]<<" "<<vec2[3]<<std::endl;
    std::cout<<std::endl;*/

   for (size_t i = 0; i < maxstep; i++)
    {
        test2.do_step(dt);
        test2.write_t_rho();
        test2.write_t_curl();
    }

    return 0;
}