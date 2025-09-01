#pragma once

#include "../pmp/surface_mesh.h"
#include "sph_gen.h"
#include "../vec3d.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>


#define PMP_SCALAR_TYPE_64

using namespace pmp;

class MUSCL_base_geometry
{

protected:
    std::vector<vector3d<double>> vertices;       // coordinates of points
    std::vector<std::vector<int>> faces;          // indexes of vertices that make a face
    std::vector<std::vector<int>> neighbors;      // indexes of neighbor faces of a given face
    std::vector<std::vector<int>> neighbors_edge; // indexes of neighbor faces of a given face that share at least an edge
    std::vector<vector3d<double>> face_centers;
    std::vector<vector3d<double>> normals;                   // normals to surface
    std::vector<std::vector<vector3d<double>>> edge_normals; // normals to edge
    std::vector<double> surface_area;

    std::vector<std::vector<std::vector<int>>> flux_faces_plus;  //+ flux face numbers for each face for each vertice
    std::vector<std::vector<std::vector<int>>> flux_faces_minus; //- flux face numbers for each face for each vertice
    std::vector<std::vector<double>> H_minus;                    // distances from the center of face to  H_minus for each face for each vertice
    std::vector<std::vector<double>> H_plus;                     // distances fromt the center of face to to H_plus for each face for each vertice

    std::vector<std::vector<double>> BM_dist; // distances from face centers to edge centers

    std::vector<std::vector<std::vector<double>>> betas_plus;  // baricentric distances to H_plus from face centers for each face for each vertice
    std::vector<std::vector<std::vector<double>>> betas_minus; // baricentric distances to H_minus from face centers for each face for each vertice

public:
    MUSCL_base_geometry(SurfaceMesh mesh)
    {

        vector3d<double> r1, r2;

        vertices.resize(mesh.n_vertices());
        flux_faces_plus.resize(mesh.n_faces());
        faces.resize(mesh.n_faces());
        flux_faces_minus.resize(mesh.n_faces());
        face_centers.resize(mesh.n_faces());
        H_minus.resize(mesh.n_faces());
        neighbors.resize(mesh.n_faces());
        neighbors_edge.resize(mesh.n_faces());
        H_plus.resize(mesh.n_faces());
        normals.resize(mesh.n_faces());
        betas_plus.resize(mesh.n_faces());
        betas_minus.resize(mesh.n_faces());
        surface_area.resize(mesh.n_faces());
        BM_dist.resize(mesh.n_faces());
        edge_normals.resize(mesh.n_faces());

        std::cout << "processing mesh..." << std::endl;

        size_t i = 0; // getting vertices from SurfaceMesh
        auto points = mesh.get_vertex_property<Point>("v:point");
        for (auto vertice : mesh.vertices())
        {
            for (size_t j = 0; j < points[vertice].size(); j++)
            {
                vertices[i][j] = points[vertice][j];
            }
            i++;
        }

        i = 0;

        for (auto face : mesh.faces()) // getting vertice numbers for faces
        {
            auto face_vertices = mesh.vertices(face);

            for (auto vertice = face_vertices.begin(); vertice != face_vertices.end(); ++vertice)
            {
                faces[i].push_back((*vertice).idx());
            }
            i++;
        }

        i = 0;
        for (auto face : mesh.faces()) // finding all neighbors for faces
        {
            auto face_vertices = mesh.vertices(face);

            for (auto vertice = face_vertices.begin(); vertice != face_vertices.end(); ++vertice)
            {
                auto faces_around = mesh.faces(*vertice);

                for (auto face_in : faces_around)
                {
                    neighbors[i].push_back((face_in).idx());
                }
            }
            i++;
        }

        i = 0;
        for (auto &&neighbor : neighbors)
        { // removing duplicates from neighbors

            std::sort(neighbor.begin(), neighbor.end());
            auto last = std::unique(neighbor.begin(), neighbor.end());
            neighbor.erase(last, neighbor.end());
            neighbor.erase(std::remove_if(neighbor.begin(), neighbor.end(), [&i](int x)
                                          { return x == i; }),
                           neighbor.end());
            i++;
        }

        /*std::vector<int> temp;
        std::vector<int> v1, v2;

        for (size_t i = 0; i < neighbors.size(); i++)
        {
            v2 = faces[i];
            std::sort(v2.begin(), v2.end());
            for (size_t j = 0; j < neighbors[i].size(); j++)
            {
                v1 = faces[neighbors[i][j]];

                std::sort(v1.begin(), v1.end());
                std::set_intersection(v2.begin(), v2.end(),
                                      v1.begin(), v1.end(),
                                      back_inserter(temp));

                if (temp.size() > 1)
                {

                    neighbors_edge[i].push_back(neighbors[i][j]);
                }
                temp.clear();
            }
        }*/
        // finding neighboors that share at least an edge in the correct order
        std::vector<int> v1;
        int f1, f2;
        for (size_t i = 0; i < faces.size(); i++)
        {

            for (size_t j = 0; j < faces[i].size(); j++)
            {
                f1 = faces[i][j];

                if (j != faces[i].size() - 1)
                {
                    f2 = faces[i][j + 1];
                }
                else
                {
                    f2 = faces[i][0];
                }

                for (size_t k = 0; k < neighbors[i].size(); k++)
                {
                    v1 = faces[neighbors[i][k]];

                    if ((std::find(v1.begin(), v1.end(), f1) != v1.end()) &&
                        (std::find(v1.begin(), v1.end(), f2) != v1.end()))
                    {
                        neighbors_edge[i].push_back(neighbors[i][k]);
                    }
                }
            }
        }

        for (size_t i = 0; i < mesh.n_faces(); i++)
        {
            face_centers[i][0] = 0;
            face_centers[i][1] = 0;
            face_centers[i][2] = 0;
            for (size_t k = 0; k < faces[i].size(); k++)
            {
                face_centers[i] += vertices[faces[i][k]];
            }

            face_centers[i] /= faces[i].size();
        }

        for (size_t i = 0; i < mesh.n_faces(); i++)
        {
            normals[i][0] = 0;
            normals[i][1] = 0;
            normals[i][2] = 0;

            for (size_t j = 0; j < faces[i].size(); j++)
            {

                if (j + 2 == faces[i].size())
                {
                    r2 = vertices[faces[i][0]] - vertices[faces[i][j + 1]];
                    r1 = vertices[faces[i][j + 1]] - vertices[faces[i][j]];
                }
                else if (j + 1 == faces[i].size())
                {
                    r1 = vertices[faces[i][0]] - vertices[faces[i][j]];
                    r2 = vertices[faces[i][1]] - vertices[faces[i][0]];
                }
                else
                {
                    r2 = vertices[faces[i][j + 2]] - vertices[faces[i][j + 1]];
                    r1 = vertices[faces[i][j + 1]] - vertices[faces[i][j]];
                }

                normals[i] += cross_product(r1, r2);
            }

            normals[i] /= normals[i].norm();
        }

        find_surface_areas();

        process_mesh();

        std::cout << "processing mesh done" << std::endl;
    };

    void process_mesh()
    {
        vector3d<double> BM, B_nB, B_nB_face, BM_normal, B_left, B_right, B1B2, r, Hm, Hp;
        vector3d<double> B_left_face, B_right_face;
        double max_cos_left = -1, max_cos_right = -1;
        int left_face1, left_face2, right_face1, right_face2, Hp_face, Hm_face;

        for (int n_face = 0; n_face < faces.size(); n_face++)
        {
            // int n_face = 1;
            std::vector<int> face = faces[n_face];
            flux_faces_minus[n_face].resize(face.size());
            betas_minus[n_face].resize(face.size());
            H_minus[n_face].resize(face.size());

            flux_faces_plus[n_face].resize(face.size());
            betas_plus[n_face].resize(face.size());
            H_plus[n_face].resize(face.size());
            edge_normals[n_face].resize(face.size());

            for (size_t i = 0; i < face.size(); i++)
            {
                max_cos_left = -1;
                max_cos_right = -1;

                vector3d<double> r1;
                if (i == face.size() - 1)
                {
                    BM = (vertices[face[i]] + vertices[face[0]]) / 2 - face_centers[n_face];
                    r = (vertices[face[i]] - vertices[face[0]]);
                    r1 = (vertices[face[i]] + vertices[face[0]]);
                    r1 /= r1.norm();
                    // edge_normals[n_face][i] = cross_product( vertices[face[0]],vertices[face[i]]); // v6
                }
                else
                {
                    BM = (vertices[face[i]] + vertices[face[i + 1]]) / 2 - face_centers[n_face];
                    r = (vertices[face[i]] - vertices[face[i + 1]]);
                    r1 = (vertices[face[i]] + vertices[face[i + 1]]);
                    r1 /= r1.norm();
                    // edge_normals[n_face][i] = cross_product(vertices[face[i+1]],vertices[face[i]]); // v6
                }

                // edge_normals[n_face][i] = cross_product(normals[n_face], r); //v1
                // edge_normals[n_face][i] = cross_product(normals[neighbors_edge[n_face][i]],r); //v2
                // edge_normals[n_face][i] = cross_product(normals[n_face], r)+cross_product(r, normals[neighbors_edge[n_face][i]]); // v3
                // edge_normals[n_face][i] = cross_product((BM + face_centers[n_face]), r); // v4

                edge_normals[n_face][i] = cross_product(r1, r); // v5

                edge_normals[n_face][i] /= edge_normals[n_face][i].norm();

                // check if normals are actually going outwards
                if ((BM + edge_normals[n_face][i] * 0.01).norm() < BM.norm())
                {
                    std::cout << "wrong edge normal direction in face: " << n_face << " on edge: " << i << std::endl;
                    edge_normals[n_face][i] *= -1;
                }

                BM_dist[n_face].push_back(distance(face_centers[n_face], r1));

                for (auto neighboor : neighbors[n_face]) // find 1st most left and most right element
                {
                    // left face
                    // B_nB = face_centers[n_face] - face_centers[neighboor]; // not projected yet
                    B_nB = face_centers[n_face] - face_centers[neighboor];
                    B_nB_face = B_nB - normals[n_face] * dot_product(B_nB, normals[n_face]); // projection

                    double cos_etha = dot_product(BM, B_nB_face) / (BM.norm() * B_nB_face.norm());
                    if (cos_etha > max_cos_left)
                    {
                        max_cos_left = cos_etha;
                        left_face1 = neighboor;
                    }

                    // right face
                    // B_nB = face_centers[neighboor] - face_centers[n_face];                   // not projected yet
                    B_nB = face_centers[neighboor] - face_centers[n_face];
                    B_nB_face = B_nB - normals[n_face] * dot_product(B_nB, normals[n_face]); // projection

                    cos_etha = dot_product(BM, B_nB_face) / (BM.norm() * B_nB_face.norm());
                    if (cos_etha > max_cos_right)
                    {
                        max_cos_right = cos_etha;
                        right_face1 = neighboor;
                    }
                }

                BM_normal = cross_product(BM, normals[n_face]);
                max_cos_left = -1;
                max_cos_right = -1;
                left_face2 = left_face1; // in case "if" doesent go off
                right_face2 = right_face1;

                B_left = face_centers[n_face] - face_centers[left_face1];   // vector in direction of B_left_1
                B_right = face_centers[right_face1] - face_centers[n_face]; // vector in direction of B_right_1

                B_left_face = B_left - normals[n_face] * dot_product(B_left, normals[n_face]);    // projection
                B_right_face = B_right - normals[n_face] * dot_product(B_right, normals[n_face]); // projection

                for (auto neighboor : neighbors[n_face]) // find second most left and right elements
                {
                    // left face
                    // B_nB = face_centers[n_face] - face_centers[neighboor];
                    B_nB = face_centers[n_face] - face_centers[neighboor];
                    B_nB_face = B_nB - normals[n_face] * dot_product(B_nB, normals[n_face]); // projection

                    double cos_etha = dot_product(BM, B_nB_face) / (BM.norm() * B_nB_face.norm());

                    // double line_cross_check = dot_product(normals[n_face], cross_product(B_nB_face, BM)) *
                    //                         dot_product(normals[n_face], cross_product(B_left_face, BM));

                    double line_cross_check = dot_product(cross_product(face_centers[n_face], r1 / 2), B_nB) *
                                              dot_product(cross_product(face_centers[n_face], r1 / 2), B_left);

                    if ((cos_etha > max_cos_left) && (neighboor != left_face1) && (line_cross_check <= 0))
                    {
                        max_cos_left = cos_etha;
                        left_face2 = neighboor;
                    }

                    // right face
                    B_nB = face_centers[neighboor] - face_centers[n_face];
                    B_nB_face = B_nB - normals[n_face] * dot_product(B_nB, normals[n_face]); // projection

                    cos_etha = dot_product(BM, B_nB_face) / (BM.norm() * B_nB_face.norm());

                    // std::cout<<cos_etha<<std::endl;
                    line_cross_check = dot_product(normals[n_face], cross_product(B_nB_face, BM)) *
                                       dot_product(normals[n_face], cross_product(B_right_face, BM));

                    if ((cos_etha > max_cos_right) && (neighboor != right_face1) && (line_cross_check <= 0))
                    {
                        max_cos_right = cos_etha;
                        right_face2 = neighboor;
                    }
                }

                vector3d<double> p3 = find_line_surf_intersection(face_centers[left_face1], face_centers[left_face2] - face_centers[left_face1],
                                                                  face_centers[n_face], BM, face_centers[n_face]);
                Hm = p3 / p3.norm();

                p3 = find_line_surf_intersection(face_centers[right_face1], face_centers[right_face2] - face_centers[right_face1],
                                                 face_centers[n_face], BM, face_centers[n_face]);

                Hp = p3 / p3.norm();

                // check to which face Hm belongs
                // is faster and safer tham point_in_face
                if (std::abs(dot_product((Hm - vertices[faces[left_face1][0]]), normals[left_face1])) <
                    std::abs(dot_product((Hm - vertices[faces[left_face2][0]]), normals[left_face2])))
                {
                    Hm_face = left_face1;
                }
                else
                {
                    Hm_face = left_face2;
                }

                if (std::abs(dot_product((Hp - vertices[faces[right_face1][0]]), normals[right_face1])) <
                    std::abs(dot_product((Hp - vertices[faces[right_face2][0]]), normals[right_face2])))
                {
                    Hp_face = right_face1;
                }
                else
                {
                    Hp_face = right_face2;
                }

                // double Hm_dist = broken_distance(face_centers[n_face], Hm, n_face, Hm_face);
                double Hm_dist = distance(face_centers[n_face], Hm);

                if (Hm_dist > 2 * (Hm - face_centers[n_face]).norm())
                {
                    std::cout << "check distances" << std::endl;
                }

                double B1B2_d = broken_distance(face_centers[left_face1], face_centers[left_face2], left_face1, left_face2);

                H_minus[n_face][i] = Hm_dist;
                flux_faces_minus[n_face][i].push_back(left_face1);
                flux_faces_minus[n_face][i].push_back(left_face2);

                double hm2 = broken_distance(face_centers[left_face2], Hm, left_face2, Hm_face);
                double hm1 = broken_distance(face_centers[left_face1], Hm, left_face1, Hm_face);

                if (std::abs((hm2 + hm1) / (distance(face_centers[left_face1], face_centers[left_face2])) - 1) > 1e-8)
                {
                    // double fd=distance(face_centers[left_face1],face_centers[left_face2]);
                    std::cout << "2 left faces are not on the same line:"
                              << n_face << " " << i << " " << hm1 / B1B2_d << " " << hm2 / B1B2_d << " "
                              << std::abs(((hm2 + hm1) / B1B2_d) - 1) << "\n";
                }




                /*if(n_face==65 && i==0){

                    B_nB = face_centers[n_face] - face_centers[left_face2];
                    B_nB_face = B_nB - normals[n_face] * dot_product(face_centers[n_face] - face_centers[left_face2], normals[n_face]); // projection


                    std::cout<<left_face1<<" "<<left_face2<<" "<<
                    dot_product(cross_product(face_centers[n_face],r1/2),  face_centers[left_face2]-face_centers[n_face]) *
                    dot_product(cross_product(face_centers[n_face],r1/2), face_centers[left_face1]-face_centers[n_face])<<"\n";

                    vertices[faces[n_face][0]].print();
                    vertices[faces[n_face][1]].print();
                    vertices[faces[n_face][2]].print();

                    vertices[faces[left_face1][0]].print();
                    vertices[faces[left_face1][1]].print();
                    vertices[faces[left_face1][2]].print();

                    vertices[faces[left_face2][0]].print();
                    vertices[faces[left_face2][1]].print();
                    vertices[faces[left_face2][2]].print();



                    face_centers[left_face1].print();
                    face_centers[left_face2].print();
                    Hm.print();


                    double fd=distance(face_centers[left_face1],face_centers[left_face2]);
                    std::cout<<n_face<<" "<<i<<" "<<hm1/fd<<" "<<hm2/fd<<" "
                    <<std::abs(((hm2+hm1)/fd)-1) <<"\n";

                }*/
                /*if(n_face==0){
                vertices[faces[0][0]].print();
                vertices[faces[0][1]].print();
                vertices[faces[0][2]].print();
                std::cout<<"\n";
                vertices[faces[left_face1][0]].print();
                vertices[faces[left_face1][1]].print();
                vertices[faces[left_face1][2]].print();
                std::cout<<"\n";
                vertices[faces[left_face2][0]].print();
                vertices[faces[left_face2][1]].print();
                vertices[faces[left_face2][2]].print();
                std::cout<<"\n";
                face_centers[left_face1].print();
                face_centers[left_face2].print();
                Hm.print();
                //find_line_surf_intersection(face_centers[left_face1], face_centers[left_face2] - face_centers[left_face1],
                //                                                  face_centers[n_face], BM, face_centers[n_face]).print();
                //std::cout<<"kekW";
                std::cout<<"\n";
                std::cout<<"\n";
                std::cout<<"\n";
                }*/



                betas_minus[n_face][i].push_back(hm2 / B1B2_d);
                betas_minus[n_face][i].push_back(hm1 / B1B2_d);

                // double Hp_dist = broken_distance(face_centers[n_face], Hp, n_face, Hp_face);
                double Hp_dist = distance(face_centers[n_face], Hp);

                if (Hp_dist > 2 * (Hp - face_centers[n_face]).norm())
                {
                    std::cout << "check distances" << std::endl;
                }

                double B1B2p_d = broken_distance(face_centers[right_face1], face_centers[right_face2], right_face1, right_face2);

                H_plus[n_face][i] = Hp_dist;
                flux_faces_plus[n_face][i].push_back(right_face1);
                flux_faces_plus[n_face][i].push_back(right_face2);

                double hp2 = broken_distance(face_centers[right_face2], Hp, right_face2, Hp_face);
                double hp1 = broken_distance(face_centers[right_face1], Hp, right_face1, Hp_face);

                betas_plus[n_face][i].push_back(hp2 / B1B2p_d);
                betas_plus[n_face][i].push_back(hp1 / B1B2p_d);

                if (std::abs((hp2 + hp1) / (distance(face_centers[right_face1], face_centers[right_face2])) - 1) > 1e-8)
                {
                    std::cout << "2 right faces are not on the same line:" << n_face << " " << i << " " << hp1 / B1B2p_d << " " << hp2 / B1B2p_d << " "
                              << std::abs(((hp2 + hp1) / B1B2p_d) - 1) << "\n";
                }

                if (std::abs(betas_plus[n_face][i][0] + betas_plus[n_face][i][1]) - 1 > 1e-8)
                {
                    std::cout << "sum of betas_plus is not 1" << std::endl;
                    std::cout << betas_plus[n_face][i][0] + betas_plus[n_face][i][1] << std::endl;
                    std::cout << n_face << " " << i << std::endl;
                }

                if (std::abs(betas_minus[n_face][i][0] + betas_minus[n_face][i][1]) - 1 > 1e-8)
                {
                    std::cout << "sum of betas_minus is not 1" << std::endl;
                    std::cout << betas_plus[n_face][i][0] + betas_plus[n_face][i][1] << std::endl;
                    std::cout << n_face << " " << i << std::endl;
                }



                /*int j1 = j0 + 1;
                int i1 = i + 1;
                if ((j0 + 1) == faces[n_face].size())
                    j1 = 0;

                if ((i + 1) == faces[n_face].size())
                    i1 = 0;

                std::cout << n_face << " " << neighbors_edge[neighboor_num][j0] << std::endl;
                std::cout << faces[n_face][i] << " " << faces[n_face][i1] << std::endl;
                std::cout << faces[neighboor_num][j0] << " " << faces[neighboor_num][j1] << std::endl;
                std::cout << std::endl;*/
            }
        }



        /*for (size_t n_face = 0; n_face < faces.size(); n_face++)
        {
            for (size_t i = 0; i < faces[n_face].size(); i++)
            {
                int neighboor_num = neighbors_edge[n_face][i];
                int j0 = (std::find(neighbors_edge[neighboor_num].begin(), neighbors_edge[neighboor_num].end(), n_face) - neighbors_edge[neighboor_num].begin());

                std::cout<<n_face<<" "<<i<<std::endl;
                edge_normals[n_face][i].print();
                std::cout<<neighboor_num<<" "<<j0<<std::endl;
                edge_normals[neighboor_num][j0].print();

            }

        }


        vertices[faces[0][0]].print();
        vertices[faces[0][1]].print();

        vertices[faces[0][2]].print();
        vertices[faces[0][3]].print();

        vertices[faces[8][2]].print();
        vertices[faces[8][3]].print();

        ((vertices[faces[0][0]] / 2 + vertices[faces[0][1]] / 2) + edge_normals[0][0] * 0.5).print();
        ((vertices[faces[8][0]] / 2 + vertices[faces[8][1]] / 2) + edge_normals[8][0] * 0.5).print();
        */
    };

    /*double broken_distance(vector3d<double> a, vector3d<double> b, int start_face, int end_face){

        return (b-a).norm();
    }*/

    /*double broken_distance(vector3d<double> a, vector3d<double> b, int start_face, int end_face)
    {
        // broken distance between 2 points if broken line made out of 2 parts
        // dim(a)=3; dim(b)=3
        // distance on a spherical mesh between points a and b

        double dist = 0;
        int maxiter = 1; // temp solution
        int iter = 0;
        // double prec = 1e-9;

        int current_face = start_face;

        vector3d<double> bma, intersection, intersection_prev;
        std::vector<int> intersect_faces;
        intersection_prev = a;

        while ((current_face != end_face) && (iter < maxiter))
        {
            // std::cout<<"a1="<<point_in_face(a)[0]<<std::endl;
            // std::cout<<current_face<<std::endl;

            bma = b - a; // line vector

            intersection = broken_distance_base(intersection_prev, b, bma, current_face);

            dist += distance(intersection, intersection_prev);

            intersect_faces = point_in_face(intersection, current_face);

            if (intersect_faces[0] == current_face)
            {
                current_face = intersect_faces[1];
            }
            else
            {
                current_face = intersect_faces[0];
            }

            intersection_prev = intersection;

            iter++;
        }

        intersection = intersection_prev;
        dist += distance(b, intersection);

        if (start_face != end_face)
        {
            bma = a - b;
            intersection = broken_distance_base(b, a, bma, end_face);
            dist += distance(a, intersection);
            dist += distance(b, intersection);
        }

        dist /= 2.;

        return dist;
    }*/

    double broken_distance(vector3d<double> a, vector3d<double> b, int start_face, int end_face)
    {

        return distance(a, b);
    }

    vector3d<double> broken_distance_base(vector3d<double> a, vector3d<double> b, vector3d<double> bma, int face_num)
    {
        // returns intersection point from a to b among projection of bma inside  face with index face_num

        double dist = 0;
        double prec = 1e-9;
        double min_etha, sinetha, cosetha;
        int edge1, edge2;
        signed int sign;
        signed int sign_prev = 0;
        vector3d<double> bma_face, r, r_edge; // b-a and (b-a) projected to face, r-- edge vector
        vector3d<double> intersection;        // rhs of linear system and intersection points
        Eigen::Matrix<double, 3, 3> dist_matrix;
        Eigen::Matrix<double, 3, 1> rhs;

        size_t n_edges = faces[face_num].size();

        bma_face = bma - normals[face_num] * dot_product(bma, normals[face_num]);
        bma_face /= bma_face.norm();

        for (size_t i = 0; i < n_edges; i++) // finding edge that our projected vector crosses
        {

            if (i < n_edges - 1)
            {
                r = (vertices[faces[face_num][i]] + vertices[faces[face_num][i + 1]]) / 2 - a;
            }
            else
            {
                r = (vertices[faces[face_num][i]] + vertices[faces[face_num][0]]) / 2 - a;
            }

            if (std::abs(dot_product(cross_product(bma_face, r), normals[face_num])) < prec)
            {

                sinetha = 0;
            }
            else
            {

                sinetha = (cross_product(bma_face, r)).norm() / (bma_face.norm() * r.norm()) *
                          dot_product(cross_product(bma_face, r), normals[face_num]) / std::abs(dot_product(cross_product(bma_face, r), normals[face_num]));
            }

            cosetha = dot_product(bma_face, r) / (bma_face.norm() * r.norm());

            if (i == 0)
            {
                min_etha = std::abs(std::atan2(sinetha, cosetha));
                edge1 = 0;
                edge2 = 1;
            }
            else if (std::abs(std::atan2(sinetha, cosetha)) < min_etha)
            {
                min_etha = std::abs(std::atan2(sinetha, cosetha));
                edge1 = i;
                edge2 = i + 1;
                if (edge2 == n_edges)
                {
                    edge2 = 0;
                }
            }

            // std::cout << dot_product(cross_product(bma_face, r), normals[face_num]) << std::endl;

            // cross_product(bma_face, r).print();
            // normals[face_num].print();
        }

        r_edge = vertices[faces[face_num][edge2]] - vertices[faces[face_num][edge1]];

        intersection = find_lines_intersection(a, vertices[faces[face_num][edge1]], bma_face, r_edge);

        return intersection;
    };

    vector3d<double> find_lines_intersection(vector3d<double> x1, vector3d<double> x2,
                                             vector3d<double> n1, vector3d<double> n2)
    {
        // x1 and x2 -- starting points of lines
        // n1 and n2 -- vectors of lines
        // method returns coordinates of point of intersection
        vector3d<double> res;
        double t0, t1, t, tau;
        double prec = 1e-9; // compare with 0

        t0 = 0;
        t1 = 0;
        // std::cout<<"here->"<<std::abs(n2[0]) <<" "<<std::abs(n2[1])<<" "<< std::abs(n2[2])<<std::endl;

        if (std::abs(n2[0]) > prec)
        {

            // std::cout<<std::abs(n1[1] - n2[1] * n1[0] / n2[0])<<" "<<std::abs(n1[2] - n2[2] * n1[0] / n2[0])<<std::endl;

            if (std::abs(n1[1] - n2[1] * n1[0] / n2[0]) > prec)
            {
                t0 = (x2[1] - x1[1] + n2[1] * (x1[0] - x2[0]) / n2[0]) / (n1[1] - n2[1] * n1[0] / n2[0]);
            }
            if (std::abs(n1[2] - n2[2] * n1[0] / n2[0]) > prec)
            {
                t1 = (x2[2] - x1[2] + n2[2] * (x1[0] - x2[0]) / n2[0]) / (n1[2] - n2[2] * n1[0] / n2[0]);
            }
            if ((std::abs(n1[1] - n2[1] * n1[0] / n2[0]) < prec) && (std::abs(n1[2] - n2[2] * n1[0] / n2[0]) < prec))
            {
                std::cout << "find_lines_intersection error: lines dont intersect" << std::endl;
                t0 = 0;
            }

            double err0 = x1[2] + n1[2] * t0 - x2[2] - n2[2] * (x1[0] - x2[0] + n1[0] * t0) / n2[0];
            double err1 = x1[1] + n1[1] * t0 - x2[1] - n2[1] * (x1[0] - x2[0] + n1[0] * t1) / n2[0];

            if (std::abs(err0) < std::abs(err1))
            {
                t = t0;
            }
            else
            {
                t = t1;
            }

            tau = (x1[0] - x2[0] + n1[0] * t) / n2[0];
        }
        else if (std::abs(n2[1]) > prec)
        {
            if (std::abs(n1[0] - n2[0] * n1[1] / n2[1]) > prec)
            {
                t0 = (x2[0] - x1[0] + n2[0] * (x1[1] - x2[1]) / n2[1]) / (n1[0] - n2[0] * n1[1] / n2[1]);
            }
            if (std::abs(n1[2] - n2[2] * n1[1] / n2[1]) > prec)
            {
                t0 = (x2[2] - x1[2] + n2[2] * (x1[1] - x2[1]) / n2[1]) / (n1[2] - n2[2] * n1[1] / n2[1]);
            }
            if (std::abs(n1[2] - n2[2] * n1[1] / n2[1]) < prec && std::abs(n1[0] - n2[0] * n1[1] / n2[1]) < prec)
            {
                std::cout << "find_lines_intersection error: lines dont intersect" << std::endl;
                t0 = 0;
                tau = 0;
            }

            double err0 = x1[2] + n1[2] * t0 - x2[2] - n2[2] * (x1[1] - x2[1] + n1[1] * t0) / n2[1];
            double err1 = x1[0] + n1[0] * t0 - x2[0] - n2[0] * (x1[1] - x2[1] + n1[1] * t1) / n2[1];

            if (std::abs(err0) < std::abs(err1))
            {
                t = t0;
            }
            else
            {
                t = t1;
            }
            tau = (x1[1] - x2[1] + n1[1] * t) / n2[1];
        }
        else if (std::abs(n2[2]) > prec)
        {
            if (std::abs(n1[0] - n2[0] * n1[2] / n2[2]) > prec)
            {
                t0 = (x2[0] - x1[0] + n2[0] * (x1[2] - x2[2]) / n2[2]) / (n1[0] - n2[0] * n1[2] / n2[2]);
            }
            if (std::abs(n1[1] - n2[1] * n1[2] / n2[2]) > prec)
            {
                t0 = (x2[1] - x1[1] + n2[1] * (x1[2] - x2[2]) / n2[2]) / (n1[1] - n2[1] * n1[2] / n2[2]);
            }
            if (std::abs(n1[1] - n2[1] * n1[2] / n2[2]) < prec && std::abs(n1[0] - n2[0] * n1[2] / n2[2]) < prec)
            {
                std::cout << "find_lines_intersection error: lines dont intersect" << std::endl;
                t = 0;
                tau = 0;
            }

            double err0 = x1[1] + n1[1] * t0 - x2[1] - n2[1] * (x1[2] - x2[2] + n1[2] * t0) / n2[2];
            double err1 = x1[0] + n1[0] * t0 - x2[0] - n2[0] * (x1[2] - x2[2] + n1[2] * t1) / n2[2];

            if (std::abs(err0) < std::abs(err1))
            {
                t = t0;
            }
            else
            {
                t = t1;
            }
            tau = (x1[2] - x2[2] + n1[2] * t) / n2[2];
        }
        else
        {
            std::cout << "find_lines_intersection error: lines dont intersect" << std::endl;
            t = 0;
            tau = 0;
        }

        for (size_t i = 0; i < 3; i++)
        {
            res[i] = x2[i] + tau * n2[i];
        }

        return res;
    }

    vector3d<double> find_line_surf_intersection(vector3d<double> xl, vector3d<double> nl,
                                                 vector3d<double> xs, vector3d<double> vs1, vector3d<double> vs2)
    {
        // xl -- line starting point, nl -- line vector
        // xs -- surface starting point,  vs1,2 -- any vector in surface(non parallel)

        vector3d<double> res;
        double A, B, C, D, t;
        double eps = 1e-10;
        if (cross_product(vs1, vs2).norm() < eps)
        {
            std::cout << "find_line_surf_intersection:parallel vectors from 1 point cannot define surface" << std::endl;
        }

        A = vs1[1] * vs2[2] - vs1[2] * vs2[1];
        B = vs1[2] * vs2[0] - vs1[0] * vs2[2];
        C = vs1[0] * vs2[1] - vs1[1] * vs2[0];
        D = -xs[0] * A - xs[1] * B - xs[2] * C;

        t = -(D + A * xl[0] + B * xl[1] + C * xl[2]) / (A * nl[0] + B * nl[1] + C * nl[2]);

        res = xl + nl * t;

        return res;
    }

    std::vector<int> point_in_face(vector3d<double> r)
    { // dim(r)=3;
        // finds face in which point r is in
        int count = 0;
        std::vector<int> res;
        int flag = 0;
        int rel_min_ind;
        double dotpr, min;
        double eps = 1e-3; // std::abs precision

        min = 10;

        for (auto face : faces)
        {

            dotpr = dot_product((r - vertices[faces[count][0]]), normals[count]);
            if (std::abs(dotpr) < eps) // compare with 0
            {
                res.push_back(count);
                flag += 1;
            }

            if (std::abs(dotpr) < min)
            {
                min = std::abs(dotpr);
                rel_min_ind = count;
            }

            count++;
        }

        if (flag == 0)
        {
            // std::cout << "point_in_face: no face has been found, returning closest" << std::endl;
            res.push_back(rel_min_ind);
        }

        // std::cout << res.size() << std::endl;

        return res;
    };

    std::vector<int> point_in_face(vector3d<double> r, int face_clue)
    {
        // dim(r)=3;
        // finds face in which point r is in
        // clue = n of face where we search for face
        int count = 0;
        std::vector<int> res;
        int flag = 0;
        int rel_min_ind;
        double dotpr, min;
        double eps = 1e-3; // std::abs precision

        min = 10;

        for (auto nei : neighbors[face_clue])
        {
            dotpr = dot_product((r - vertices[faces[nei][0]]), normals[nei]);
            if (std::abs(dotpr) < eps) // compare with 0
            {
                res.push_back(nei);
                flag += 1;
            }

            if (std::abs(dotpr) < min)
            {
                min = std::abs(dotpr);
                rel_min_ind = count;
            }
        }

        if (flag == 0)
        {
            // std::cout << "point_in_face: no face has been found, returning closest "<<min << std::endl;
            res.push_back(rel_min_ind);
        }

        return res;
    };

    bool is_on_surface(vector3d<double> r)
    {
        int count = 0;
        bool res = false;
        double dotpr;
        vector3d<double> r_face;
        double eps = 1e-3; // std::abs precision

        for (auto face : faces)
        {
            r_face = face_centers[count];
            dotpr = dot_product((r_face - r), normals[count]);
            if (std::abs(dotpr) < eps) // compare with 0
            {
                res = true;
            }

            count++;
        }

        return res;
    }

    /*double distance(vector3d<double> a,vector3d<double> b)//straight line
    {
        return (a-b).norm();
    }*/

    double distance(vector3d<double> a, vector3d<double> b) // great circle distance
    {
        a /= a.norm();
        b /= b.norm(); // unit sphere
        double d = (a - b).norm();
        return 2 * std::asin(d / 2);
    }

    void find_surface_areas()
    {
        int j1;
        double S_total = 0;

        double a, b, c; // sph triangle sides in radians
        double A, B, C; // angles of sph triangles

        for (size_t i = 0; i < faces.size(); i++)
        {
            surface_area[i] = 0;

            for (size_t j = 0; j < faces[i].size(); j++)
            {

                if (j != faces[i].size() - 1)
                {
                    j1 = j + 1;
                }
                else
                {
                    j1 = 0;
                }
                a = distance(face_centers[i] / face_centers[i].norm(), vertices[faces[i][j]]); // distance = arc length on a unit sphere
                b = distance(face_centers[i] / face_centers[i].norm(), vertices[faces[i][j1]]);
                c = distance(vertices[faces[i][j]], vertices[faces[i][j1]]);

                A = std::acos((std::cos(a) - std::cos(b) * std::cos(c)) / (std::sin(b) * std::sin(c)));
                B = std::acos((std::cos(b) - std::cos(a) * std::cos(c)) / (std::sin(a) * std::sin(c)));
                C = std::acos((std::cos(c) - std::cos(b) * std::cos(a)) / (std::sin(b) * std::sin(a)));

                surface_area[i] += A + B + C - M_PI;
                if(surface_area[i]<0){
                    std::cout<<"check surface areas, S < 0 detected! \n";
                }

                S_total += A + B + C - M_PI;

                //surface_area[i] += (cross_product(face_centers[i] - vertices[faces[i][j]], face_centers[i] - vertices[faces[i][j1]])).norm() / 2.;
            }
        }
    }

    void print_vertices()
    {

        for (auto vertice : vertices)
        {
            std::cout << "(";
            for (auto vertice_el : vertice)
            {
                std::cout << vertice_el;

                if (vertice_el != *(vertice.end() - 1))
                {
                    std::cout << ",";
                }
                else
                {
                    std::cout << ")";
                }
            }

            std::cout << std::endl;
        }
    };

    void print_neighbors()
    {
        for (auto neighbor : neighbors)
        {
            std::cout << "|";
            for (auto neighbor_element : neighbor)
            {
                std::cout << neighbor_element << "|";
            }
            std::cout << std::endl;
        }
    };

    void write_face_centers()
    {
        std::ofstream outfile;
        outfile.open("results/face_centers.dat", std::ios::out);

        for (auto face_center : face_centers)
        {
            outfile << std::setprecision(15) << face_center[0]/face_center.norm() 
            << " " << face_center[1]/face_center.norm()  << " " << face_center[2]/face_center.norm();
            outfile << "\n";
        }
        outfile.close();
    };

    vector3d<double> face_center(int n_face)
    {

        return face_centers[n_face] / face_centers[n_face].norm();
    }

    void write_faces()
    {
        std::ofstream outfile;
        outfile.open("results/faces.dat", std::ios::out);

        for (auto face : faces)
        {
            for (auto face_el : face)
                outfile << face_el << " ";

            outfile << "\n";
        }
        outfile.close();
    };

    void write_vertices()
    {
        std::ofstream outfile;
        outfile.open("results/vertices.dat", std::ios::out);

        for (auto vertice : vertices)
        {
            for (auto vertice_el : vertice)
                outfile << std::setprecision(15) << vertice_el << " ";

            outfile << "\n";
        }
        outfile.close();
    };

    void print_neighbors_edge()
    {
        for (auto neighbor : neighbors_edge)
        {
            std::cout << "|";
            for (auto neighbor_element : neighbor)
            {
                std::cout << neighbor_element << "|";
            }
            std::cout << std::endl;
        }
    };

    void print_normals()
    {
        for (auto normal : normals)
        {
            std::cout << "(" << normal[0] << "|" << normal[1] << "|" << normal[2] << ")";
            std::cout << std::endl;
        }
    };

    int n_faces()
    {

        return faces.size();
    }

    int n_vertices()
    {

        return vertices.size();
    }

protected:
    vector3d<double> cross_product(vector3d<double> a, vector3d<double> b)
    {
        vector3d<double> res;
        res[0] = a[1] * b[2] - a[2] * b[1]; // cross prod
        res[1] = a[2] * b[0] - a[0] * b[2];
        res[2] = a[0] * b[1] - a[1] * b[0];

        return res;
    }

    double dot_product(vector3d<double> a, vector3d<double> b)
    {
        double res;
        res = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        return res;
    }
};