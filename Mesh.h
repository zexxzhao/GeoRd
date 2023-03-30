#ifndef __MESH_H__
#define __MESH_H__

#include "Graph.h"
#include "Point.h"
#include <vector>

namespace GeoRd {
namespace Element {
template <typename T, T D = T{}, T t0 = T{}, T t1 = T{}, T t2 = T{}>
struct ElementProperty {
    static constexpr T dim = D;
    static constexpr T n_vertices = t0;
    static constexpr T n_edges = t1;
    static constexpr T n_faces = t2;
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

template <typename Elem, int D = Element::dim,
          typename Ownership = Layout::Exclusive>
struct Mesh {
    using Element = Elem;
    using Node = Point<double, D>;

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

template <typename Elem>
struct Mesh<Elem, Layout::Distributed> : public struct Mesh<Elem> {
    using Element = Elem;
    using Node = Point<double, Elememt::dim>;

    std::vector<std::size_t> vertex_local2global;
    std::vector<std::size_t> element_local2global;
};

template <typename Elem>
void get_vertex_connectivity_in_local_patch(
    const Mesh<Elem, Layout::Distributed> &mesh, Graph &graph) {
    graph.clear();
    for (std::size_t i = 0; i < mesh.elements.size(); ++i) {
        std::array<std::size_t, Elem::n_vertices> global_vertex_ids;
        // get global vertex ids
        for (std::size_t j = 0; j < Elem::n_vertices; ++j) {
            global_vertex_ids[j] =
                mesh.vertex_local2global[mesh.elements[i * Elem::n_vertices +
                                                       j]];
        }
        // add to graph
        for (std::size_t j = 0; j < Elem::n_vertices; ++j) {
            auto &neighbors = graph[global_vertex_ids[j]];
            for (std::size_t k = 0; k < Elem::n_vertices; ++k) {
                if (j != k) {
                    neighbors.push_back(global_vertex_ids[k]);
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

#include "MPI.h"
template <typename Elem>
void get_vertex_connectivity_in_global_patch(
    const Mesh<Elem, Layout::Distributed> &mesh, Graph &graph) {
    get_vertex_connectivity_in_local_patch(mesh, graph);

    // serialize graph
    std::vector<size_t> data;
    serialize_graph(graph, data);
    // gather data
    std::vector<size_t> recv_data;
    details::MPI_Allgather(MPI_COMM_WORLD, data, recv_data);
    // deserialize graph
    deserialize_graph(recv_data, graph);
    // sort and remove duplicates
    for (auto &kv : graph) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()),
                        kv.second.end());
    }
}

} // namespace GeoRd

#endif // __MESH_H__