#pragma once
#include "../pmp/algorithms/subdivision.h"
#define PMP_SCALAR_TYPE_64
using namespace pmp;

SurfaceMesh tetrahedron()
{
    SurfaceMesh mesh;

    // choose coordinates on the unit sphere
    double a = 1./ 3.;
    double b = sqrt(8. / 9.);
    double c = sqrt(2. / 9.);
    double d = sqrt(2. / 3.);

    // add the 4 vertices
    auto v0 = mesh.add_vertex(Point(0, 0, 1));
    auto v1 = mesh.add_vertex(Point(-c, d, -a));
    auto v2 = mesh.add_vertex(Point(-c, -d, -a));
    auto v3 = mesh.add_vertex(Point(b, 0, -a));

    // add the 4 faces
    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v2, v3);
    mesh.add_triangle(v0, v3, v1);
    mesh.add_triangle(v3, v2, v1);

    return mesh;
}

SurfaceMesh hexahedron()
{
    SurfaceMesh mesh;

    // choose coordinates on the unit sphere
    double a = 1. / sqrt(3.);

    // add the 8 vertices
    auto v0 = mesh.add_vertex(Point(-a, -a, -a));
    auto v1 = mesh.add_vertex(Point(a, -a, -a));
    auto v2 = mesh.add_vertex(Point(a, a, -a));
    auto v3 = mesh.add_vertex(Point(-a, a, -a));
    auto v4 = mesh.add_vertex(Point(-a, -a, a));
    auto v5 = mesh.add_vertex(Point(a, -a, a));
    auto v6 = mesh.add_vertex(Point(a, a, a));
    auto v7 = mesh.add_vertex(Point(-a, a, a));

    // add the 6 faces
    mesh.add_quad(v3, v2, v1, v0);
    mesh.add_quad(v2, v6, v5, v1);
    mesh.add_quad(v5, v6, v7, v4);
    mesh.add_quad(v0, v4, v7, v3);
    mesh.add_quad(v3, v7, v6, v2);
    mesh.add_quad(v1, v5, v4, v0);

    return mesh;
}

Point centroid(const SurfaceMesh &mesh, Face f)
{
    Point c(0, 0, 0);
    Scalar n(0);
    for (auto v : mesh.vertices(f))
    {
        c += mesh.position(v);
        ++n;
    }
    c /= n;
    return c;
}

void dual(SurfaceMesh &mesh)
{
    // the new dual mesh
    SurfaceMesh tmp;

    // a property to remember new vertices per face
    auto fvertex = mesh.add_face_property<Vertex>("f:vertex");

    // for each face add the centroid to the dual mesh
    for (auto f : mesh.faces())
        fvertex[f] = tmp.add_vertex(centroid(mesh, f));

    // add new face for each vertex
    for (auto v : mesh.vertices())
    {
        std::vector<Vertex> vertices;
        for (auto f : mesh.faces(v))
            vertices.push_back(fvertex[f]);

        tmp.add_face(vertices);
    }

    // swap old and new meshes, don't copy properties
    mesh.assign(tmp);
}

SurfaceMesh octahedron()
{
    auto mesh = hexahedron();
    dual(mesh);
    return mesh;
}

SurfaceMesh square(size_t n_size)
{ // a = size of sqare
    SurfaceMesh mesh;

    double dx = 1. / (n_size-1);

    // adding vertices
    for (size_t i = 0; i < n_size; i++)
    {
        for (size_t j = 0; j < n_size; j++)
        {
            mesh.add_vertex(Point(i * dx, j * dx, -sqrt(i*i+j*j)*dx/100));
        }
    }

    // grouping vertices into quads
    for (int i = 0; i < n_size-1; i++)
    {
        size_t i0 = i * n_size;
        size_t i1 = (i + 1) * n_size ;

        for (size_t j = 0; j < n_size-1; j++)
        {

            size_t j1 = i0 + j;
            size_t j0 = i0 + (j + 1) % n_size;
            size_t j3 = i1 + (j + 1) % n_size;
            size_t j2 = i1 + j;
            mesh.add_quad(Vertex(j0), Vertex(j1),Vertex(j2), Vertex(j3));
        }
    }

    return mesh;
}

void project_to_unit_sphere(SurfaceMesh &mesh)
{
    for (auto v : mesh.vertices())
    {
        auto p = mesh.position(v);
        auto n = norm(p);
        mesh.position(v) = (1. / n) * p;
    }
}

SurfaceMesh octahedron2()
{
    auto mesh = hexahedron();
    dual(mesh);
    project_to_unit_sphere(mesh);
    return mesh;
}

SurfaceMesh icosahedron()
{
    SurfaceMesh mesh;

    double phi = (1. + sqrt(5.)) * 0.5; // golden ratio
    double a = 1.0;
    double b = 1.0 / phi;

    // add vertices
    auto v1 = mesh.add_vertex(Point(0, b, -a));
    auto v2 = mesh.add_vertex(Point(b, a, 0));
    auto v3 = mesh.add_vertex(Point(-b, a, 0));
    auto v4 = mesh.add_vertex(Point(0, b, a));
    auto v5 = mesh.add_vertex(Point(0, -b, a));
    auto v6 = mesh.add_vertex(Point(-a, 0, b));
    auto v7 = mesh.add_vertex(Point(0, -b, -a));
    auto v8 = mesh.add_vertex(Point(a, 0, -b));
    auto v9 = mesh.add_vertex(Point(a, 0, b));
    auto v10 = mesh.add_vertex(Point(-a, 0, -b));
    auto v11 = mesh.add_vertex(Point(b, -a, 0));
    auto v12 = mesh.add_vertex(Point(-b, -a, 0));

    project_to_unit_sphere(mesh);

    // add triangles
    mesh.add_triangle(v3, v2, v1);
    mesh.add_triangle(v2, v3, v4);
    mesh.add_triangle(v6, v5, v4);
    mesh.add_triangle(v5, v9, v4);
    mesh.add_triangle(v8, v7, v1);
    mesh.add_triangle(v7, v10, v1);
    mesh.add_triangle(v12, v11, v5);
    mesh.add_triangle(v11, v12, v7);
    mesh.add_triangle(v10, v6, v3);
    mesh.add_triangle(v6, v10, v12);
    mesh.add_triangle(v9, v8, v2);
    mesh.add_triangle(v8, v9, v11);
    mesh.add_triangle(v3, v6, v4);
    mesh.add_triangle(v9, v2, v4);
    mesh.add_triangle(v10, v3, v1);
    mesh.add_triangle(v2, v8, v1);
    mesh.add_triangle(v12, v10, v7);
    mesh.add_triangle(v8, v11, v7);
    mesh.add_triangle(v6, v12, v5);
    mesh.add_triangle(v11, v9, v5);

    return mesh;
}

SurfaceMesh dodecahedron()
{
    auto mesh = icosahedron();
    dual(mesh);
    project_to_unit_sphere(mesh);
    return mesh;
}

SurfaceMesh uv_sphere(int n_slices, int n_stacks)
{
    SurfaceMesh mesh;

    // add top vertex
    auto v0 = mesh.add_vertex(Point(0, 1, 0));

    // generate vertices per stack / slice
    for (int i = 0; i < n_stacks - 1; i++)
    {
        auto phi = M_PI * double(i + 1) / double(n_stacks);
        for (int j = 0; j < n_slices; j++)
        {
            auto theta = 2.0 * M_PI * double(j) / double(n_slices);
            auto x = std::sin(phi) * std::cos(theta);
            auto y = std::cos(phi);
            auto z = std::sin(phi) * std::sin(theta);
            mesh.add_vertex(Point(x, y, z));
        }
    }

    // add bottom vertex
    auto v1 = mesh.add_vertex(Point(0, -1, 0));

    // add top / bottom triangles
    for (int i = 0; i < n_slices; ++i)
    {
        auto i0 = i + 1;
        auto i1 = (i + 1) % n_slices + 1;
        mesh.add_triangle(v0, Vertex(i1), Vertex(i0));
        i0 = i + n_slices * (n_stacks - 2) + 1;
        i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 1;
        mesh.add_triangle(v1, Vertex(i0), Vertex(i1));
    }

    // add quads per stack / slice
    for (int j = 0; j < n_stacks - 2; j++)
    {
        auto j0 = j * n_slices + 1;
        auto j1 = (j + 1) * n_slices + 1;
        for (int i = 0; i < n_slices; i++)
        {
            auto i0 = j0 + i;
            auto i1 = j0 + (i + 1) % n_slices;
            auto i2 = j1 + (i + 1) % n_slices;
            auto i3 = j1 + i;
            mesh.add_quad(Vertex(i0), Vertex(i1), Vertex(i2), Vertex(i3));
        }
    }
    return mesh;
}



SurfaceMesh uv_sphere_2(int n_slices, int n_stacks)
{
    SurfaceMesh mesh;

    // Goal: identical connectivity and winding as uv_sphere(), but with pole axis
    // rotated from Y to Z (swap y<->z in the uv_sphere() vertex formula).
    // MUSCL_geometry assumes a consistent winding to get outward normals and
    // positive spherical areas.

    // add top vertex (north pole, +Z). This is the rotated version of (0, +Y, 0).
    auto v0 = mesh.add_vertex(Point(0, 0, 1));

    // generate vertices per stack / slice
    // Use *exactly* the same (phi, theta) loops and ordering as uv_sphere().
    // Start with uv_sphere() coordinates:
    //   x = sin(phi) * cos(theta)
    //   y = cos(phi)
    //   z = sin(phi) * sin(theta)
    // Then rotate by swapping y<->z:
    //   (x, y, z)_new = (x, z, y)
    for (int i = 0; i < n_stacks - 1; i++)
    {
        const auto phi = M_PI * double(i + 1) / double(n_stacks);
        for (int j = 0; j < n_slices; j++)
        {
            const auto theta = 2.0 * M_PI * double(j) / double(n_slices);
            const auto x = std::sin(phi) * std::cos(theta);
            const auto y = std::sin(phi) * std::sin(theta);
            const auto z = std::cos(phi);
            mesh.add_vertex(Point(x, y, z));
        }
    }

    // add bottom vertex (south pole, -Z). Rotated version of (0, -Y, 0).
    auto v1 = mesh.add_vertex(Point(0, 0, -1));

    // add top / bottom triangles (identical indexing and winding as uv_sphere)
    for (int i = 0; i < n_slices; ++i)
    {
        auto i0 = i + 1;
        auto i1 = (i + 1) % n_slices + 1;
    // reverse winding vs uv_sphere(): rotation changes handedness (det=-1)
    mesh.add_triangle(v0, Vertex(i0), Vertex(i1));
        i0 = i + n_slices * (n_stacks - 2) + 1;
        i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 1;
    mesh.add_triangle(v1, Vertex(i1), Vertex(i0));
    }

    // add quads per stack / slice
    for (int j = 0; j < n_stacks - 2; j++)
    {
        auto j0 = j * n_slices + 1;
        auto j1 = (j + 1) * n_slices + 1;
        for (int i = 0; i < n_slices; i++)
        {
            auto i0 = j0 + i;
            auto i1 = j0 + (i + 1) % n_slices;
            auto i2 = j1 + (i + 1) % n_slices;
            auto i3 = j1 + i;
            // reverse winding
            mesh.add_quad(Vertex(i0), Vertex(i3), Vertex(i2), Vertex(i1));
        }
    }

    return mesh;
}


SurfaceMesh icosphere(size_t n_subdivisions)
{
    auto mesh = icosahedron();
    for (size_t i = 0; i < n_subdivisions; i++)
    {
        loop_subdivision(mesh);
        project_to_unit_sphere(mesh);
    }
    return mesh;
}

SurfaceMesh icosphere_hex(size_t n_subdivisions)
{
    auto mesh = icosphere(n_subdivisions);
    dual(mesh);
    project_to_unit_sphere(mesh);
    return mesh;
}

SurfaceMesh quad_sphere(size_t n_subdivisions)
{
    auto mesh = hexahedron();
    for (size_t i = 0; i < n_subdivisions; i++)
    {
        catmull_clark_subdivision(mesh);
        project_to_unit_sphere(mesh);
    }
    return mesh;
}
