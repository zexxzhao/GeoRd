#ifndef __TEST_MESH_H__
#define __TEST_MESH_H__

#include "gtest-mpi-listener.h"
#include "../include/GeoRd.h"

using namespace GeoRd;

template<typename T, typename U,
         typename std::enable_if<std::is_integral<typename std::common_type<T, U>::type>::value, int>::type = 0>
bool cmp_vector(std::vector<T> a, std::vector<U> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}
void generate_exclusive_triangle_mesh(TriangleMesh &mesh) {
    // Generate 4 points for a unit square at z = 0
    std::vector<Point3D> points;
    points.emplace_back(0.0, 0.0, 0.0);
    points.emplace_back(1.0, 0.0, 0.0);
    points.emplace_back(1.0, 1.0, 0.0);
    points.emplace_back(0.0, 1.0, 0.0);
    // Generate 2 triangles
    std::vector<std::vector<int>> cells;
    cells.emplace_back(std::vector<int>{0, 1, 2});
    cells.emplace_back(std::vector<int>{0, 2, 3});
    // Generate mesh
    mesh.vertices = points;
    mesh.elements.reserve(cells.size() * 3);
    for(const auto &cell : cells) {
        mesh.elements.insert(mesh.elements.end(), cell.begin(), cell.end());
    }
}

TEST(Mesh, ExclusiveTriangle) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();
    if (rank) return;
    TriangleMesh mesh;
    generate_exclusive_triangle_mesh(mesh);
    Graph g;
    get_vertex_connectivity(mesh, g);
    std::vector<std::size_t> expected;
    // expected 0 is connected to [1, 2, 3]
    expected = {1, 2, 3};
    ASSERT_TRUE(cmp_vector(g[0], expected));
    // expected 1 is connected to [0, 2]
    expected = {0, 2};
    ASSERT_TRUE(cmp_vector(g[1], expected));
    // expected 2 is connected to [0, 1, 3]
    expected = {0, 1, 3};
    ASSERT_TRUE(cmp_vector(g[2], expected));
    // expected 3 is connected to [0, 2]
    expected = {0, 2};
    ASSERT_TRUE(cmp_vector(g[3], expected));
}


void generate_distributed_tetrahedron_mesh(TetrahedronMesh &mesh) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();
    // Generate 8 points for a unit cube
    std::vector<Point3D> points;
    points.emplace_back(0.0, 0.0, 0.0);
    points.emplace_back(1.0, 0.0, 0.0);
    points.emplace_back(1.0, 1.0, 0.0);
    points.emplace_back(0.0, 1.0, 0.0);
    points.emplace_back(0.0, 0.0, 1.0);
    points.emplace_back(1.0, 0.0, 1.0);
    points.emplace_back(1.0, 1.0, 1.0);
    points.emplace_back(0.0, 1.0, 1.0);
    // Generate 5 tetrahedrons
    std::vector<std::vector<int>> cells;
    cells.emplace_back(std::vector<int>{0, 1, 3, 4});
    cells.emplace_back(std::vector<int>{1, 2, 3, 6});
    cells.emplace_back(std::vector<int>{1, 3, 4, 6});
    cells.emplace_back(std::vector<int>{1, 4, 5, 6});
    cells.emplace_back(std::vector<int>{3, 4, 6, 7});
    // Parition the mesh into 4 parts: element_belong_to[i] = rank
    std::vector<int> element_belong_to = {0, 1, 2, 3, 0};
    // Generate vertex local to global map
    std::vector<int> vertex_local_to_global;
    std::set<int> owned_vertices;
    for(int i = 0; i < element_belong_to.size(); ++i) {
        if(element_belong_to[i] == rank) {
            owned_vertices.insert(cells[i].begin(), cells[i].end());
        }
    }
    vertex_local_to_global.assign(owned_vertices.begin(), owned_vertices.end());
    // Generate vertex global to local map
    std::map<int, int> vertex_global_to_local;
    for(int i = 0; i < vertex_local_to_global.size(); ++i) {
        vertex_global_to_local[vertex_local_to_global[i]] = i;
    }
    // assign vertex into mesh
    auto &vert = mesh.vertices;
    vert.resize(vertex_local_to_global.size());
    for(int i = 0; i < vertex_local_to_global.size(); ++i) {
        vert[i] = points[vertex_local_to_global[i]];
    }
    // assign cells into mesh
    auto &cell = mesh.elements;
    auto n_owned_cells = std::count(element_belong_to.begin(), element_belong_to.end(), rank);
    cell.reserve(n_owned_cells * 4);
    for(int i = 0; i < element_belong_to.size(); ++i) {
        if(element_belong_to[i] == rank) {
            for(int j = 0; j < 4; ++j) {
                cell.push_back(vertex_global_to_local[cells[i][j]]);
            }
        }
    }
    // assign vertex_local_to_global into mesh
    mesh.vertex_local2global.assign(vertex_local_to_global.begin(), vertex_local_to_global.end());
}

inline void print_distributed_tetrahedron_mesh(const TetrahedronMesh &mesh) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();
    for(int i = 0; i < size; ++i) {
        if(i == rank) {
            std::cout << "Rank " << rank << ":\n";
            std::cout << "Vertices:\n";
            for(int j = 0; j < mesh.vertices.size(); ++j) {
                std::cout << mesh.vertices[j][0] << " "
                          << mesh.vertices[j][1] << " "
                          << mesh.vertices[j][2] << "\n";
            }
            std::cout << "Elements:\n";
            const auto l2g = mesh.vertex_local2global;
            for(int j = 0; j < mesh.elements.size(); j += 4) {
                std::cout << l2g[mesh.elements[j]] << " " << l2g[mesh.elements[j + 1]] << " " << l2g[mesh.elements[j + 2]] << " " << l2g[mesh.elements[j + 3]] << "\n";
            }
            std::cout << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
TEST(Mesh, DistributedTetrahedron) {
    TetrahedronMesh mesh;
    generate_distributed_tetrahedron_mesh(mesh);
    //print_distributed_tetrahedron_mesh(mesh);
    Graph graph;
    get_vertex_connectivity(mesh, graph);
    std::vector<std::size_t> expected;
    // expected 0 is connected to [1, 3, 4]
    expected = {1, 3, 4};
    ASSERT_TRUE(cmp_vector(graph[0], expected));
    // expected 1 is connected to [0, 2, 3, 4, 5, 6]
    expected = {0, 2, 3, 4, 5, 6};
    ASSERT_TRUE(cmp_vector(graph[1], expected));
    // expected 2 is connected to [1, 3, 6]
    expected = {1, 3, 6};
    ASSERT_TRUE(cmp_vector(graph[2], expected));
    // expected 3 is connected to [0, 1, 2, 4, 6, 7]
    expected = {0, 1, 2, 4, 6, 7};
    ASSERT_TRUE(cmp_vector(graph[3], expected));
    // expected 4 is connected to [0, 1, 3, 5, 6, 7]
    expected = {0, 1, 3, 5, 6, 7};
    ASSERT_TRUE(cmp_vector(graph[4], expected));
    // expected 5 is connected to [1, 4, 6]
    expected = {1, 4, 6};
    ASSERT_TRUE(cmp_vector(graph[5], expected));
    // expected 6 is connected to [1, 2, 3, 4, 5, 7]
    expected = {1, 2, 3, 4, 5, 7};
    ASSERT_TRUE(cmp_vector(graph[6], expected));
    // expected 7 is connected to [3, 4, 6]
    expected = {3, 4, 6};
    ASSERT_TRUE(cmp_vector(graph[7], expected));
}
#endif // __TEST_MESH_H__