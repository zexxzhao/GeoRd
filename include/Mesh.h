#ifndef __MESH_H__
#define __MESH_H__

#include "Graph.h"
#include "Point.h"
#include <array>
#include <vector>

namespace GeoRd {
namespace Element {
template <typename T, T D = T{}, T t0 = T{}, T t1 = T{}, T t2 = T{}>
struct ElementProperty {
    static constexpr T dim = D;
    static constexpr T n_vertices = t0;
    static constexpr T n_edges = t1;
    static constexpr T n_faces = t2;
    template<typename U>
    using VertexArray = std::array<U, n_vertices>;
};

using Point = ElementProperty<int, 0, 1, 0, 0>;

using Line = ElementProperty<int, 1, 2, 1, 0>;

using Triangle = ElementProperty<int, 2, 3, 3, 1>;
using Quadrangle = ElementProperty<int, 2, 4, 4, 1>;

using Tetrahedron = ElementProperty<int, 3, 4, 6, 4>;
using Hexahedron = ElementProperty<int, 3, 8, 12, 6>;
using Prism = ElementProperty<int, 3, 6, 9, 5>;
using Pyramid = ElementProperty<int, 3, 5, 8, 5>;
} // namespace Element

namespace Layout {
struct Exclusive {};   // default
struct Distributed {}; // MPI-based distributed mesh
} // namespace Layout

template <typename Elem, int D = Elem::dim,
          typename Ownership = Layout::Exclusive>
struct Mesh {
    using Element = Elem;
    using Node = Point<D>;
    static constexpr bool is_shared = false;

    std::vector<Node> vertices;
    std::vector<std::size_t> elements;

    void get_cell_coordinates(std::size_t cell_id,
                              std::vector<Node> &coords) const {
        coords.resize(Element::n_vertices);
        for (std::size_t i = 0; i < Element::n_vertices; ++i) {
            coords[i] = vertices[elements[cell_id * Element::n_vertices + i]];
        }
    }
};


template <typename Mesh, typename std::enable_if<not Mesh::is_shared, int>::type = 0>
void get_vertex_connectivity(const Mesh &mesh, Graph &graph) {
    graph.clear();
    using Elem = typename Mesh::Element;
    for (std::size_t i = 0; i < mesh.elements.size() / Elem::n_vertices; ++i) {
        for (std::size_t j = 0; j < Elem::n_vertices; ++j) {
            auto key = mesh.elements[i * Elem::n_vertices + j];
            auto &neighbors = graph[key];
            for (std::size_t k = 0; k < Elem::n_vertices; ++k) {
                if (j != k) {
                    neighbors.push_back(
                        mesh.elements[i * Elem::n_vertices + k]);
                }
            }
        }
    }
    // sort and remove duplicates
    for (auto &kv : graph) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()),
                        kv.second.end());
    }
}


template <typename Elem, int D>
struct Mesh<Elem, D, Layout::Distributed> : public Mesh<Elem, D> {
    using Element = Elem;
    using Node = Point<D>;
    using ExclusiveMesh = Mesh<Elem, D, Layout::Exclusive>;
    static constexpr int dim = D;
    static constexpr bool is_shared = true;

    std::vector<std::size_t> vertex_local2global;
    std::vector<std::size_t> element_local2global;
};


template <typename Mesh, typename std::enable_if<Mesh::is_shared, int>::type = 0>
void get_vertex_connectivity_in_local_patch(
    const Mesh &mesh, Graph &graph) {
    graph.clear();
    Graph local_graph;
    get_vertex_connectivity<typename Mesh::ExclusiveMesh>(mesh, local_graph);
    for (auto &kv : local_graph) {
        auto &neighbors = graph[mesh.vertex_local2global[kv.first]];
        neighbors.assign(kv.second.begin(), kv.second.end());
        std::for_each(
            neighbors.begin(), neighbors.end(),
            [&mesh](std::size_t &v) { v = mesh.vertex_local2global[v]; });
    }
}

#include "MPI.h"
template <typename Mesh, typename std::enable_if<Mesh::is_shared, int>::type = 0>
void get_vertex_connectivity_in_global_patch(
    const Mesh &mesh, Graph &graph) {
    get_vertex_connectivity_in_local_patch(mesh, graph);
    // serialize graph
    std::vector<size_t> data;
    serialize_graph(graph, data);
    // gather data
    std::vector<size_t> recv_data;
    details::MPI_allgather(MPI_COMM_WORLD, data, recv_data);
    // deserialize graph
    deserialize_graph(recv_data, graph);
    // sort and remove duplicates
    for (auto &kv : graph) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()),
                        kv.second.end());
    }
}

template <typename Mesh, typename std::enable_if<Mesh::is_shared, int>::type = 0>
void get_vertex_connectivity(const Mesh &mesh, Graph &graph) {
    get_vertex_connectivity_in_global_patch(mesh, graph);
}


} // namespace GeoRd

#endif // __MESH_H__