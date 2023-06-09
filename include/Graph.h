#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <algorithm>
#include <functional>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>
namespace GeoRd {

// Undirected graph
template <typename Vertex = std::size_t>
using DirectedGraph = std::unordered_map<Vertex, std::vector<Vertex>>;

using Graph = DirectedGraph<std::size_t>;

template <typename Vertex = std::size_t,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void add_edge(Graph &graph, Vertex vertex_id, Vertex neighbor_id) {
    auto &neighbors = graph[vertex_id];
    if (std::find(neighbors.begin(), neighbors.end(), neighbor_id) ==
        neighbors.end()) {
        neighbors.push_back(neighbor_id);
    }
}

template <typename Vertex = std::size_t,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void add_edge(Graph &graph, Vertex vertex_id,
              const std::vector<Vertex> &neighbor_ids) {
    for (const auto &neighbor_id : neighbor_ids) {
        add_edge(graph, vertex_id, neighbor_id);
    }
}

// serialize graph to a std::vector
template <typename Vertex = std::size_t,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void serialize_graph(const Graph &graph, std::vector<Vertex> &data) {
    data.clear();
    auto n_edges = std::accumulate(
        graph.begin(), graph.end(), 0,
        [](std::size_t acc, const auto &kv) { return acc + kv.second.size(); });
    data.reserve(n_edges + graph.size() * 2);
    for (const auto &kv : graph) {
        // push back the length of the edge list
        data.push_back(kv.second.size());
        // push back the vertex id
        data.push_back(kv.first);
        // push back the edge list
        data.insert(data.end(), kv.second.begin(), kv.second.end());
    }
}

// deserialize graph from a std::vector
template <typename Vertex = std::size_t,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void deserialize_graph(const std::vector<Vertex> &data, Graph &graph) {
    graph.clear();
    auto it = data.begin();
    while (it != data.end()) {
        auto n_edges = *it++;
        auto vertex_id = *it++;
        auto &neighbors = graph[vertex_id];
        neighbors.reserve(neighbors.size() + n_edges);
        neighbors.insert(neighbors.end(), it, it + n_edges);
        it += n_edges;
    }
}

// non recursive depth first search with additional restriction
template <typename Vertex,
          typename Callable = std::function<bool(Vertex, Vertex)>,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void dfs(
    const Graph &graph, Vertex vertex_id, std::vector<Vertex> &visited,
    Callable &&additional_restriction = [](Vertex, Vertex) { return true; }) {
    if (graph.empty())
        return;
    std::vector<Vertex> stack;
    stack.push_back(vertex_id);
    while (!stack.empty()) {
        auto vertex_id = stack.back();
        stack.pop_back();
        if (std::find(visited.begin(), visited.end(), vertex_id) ==
            visited.end()) {
            visited.push_back(vertex_id);
            for (auto neighbor_id : graph.at(vertex_id)) {
                if (additional_restriction(vertex_id, neighbor_id)) {
                    stack.push_back(neighbor_id);
                }
            }
        }
    }
}

// traverse the graph and collect connected components using DFS
template <typename Vertex,
          typename Callable = std::function<bool(Vertex, Vertex)>,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void connected_components(
    const Graph &graph, std::vector<std::vector<Vertex>> &components,
    Callable &&additional_restriction = [](Vertex, Vertex) { return true; }) {
    if (graph.empty())
        return;
    std::vector<bool> visited(graph.size(), 0);
    for (const auto &kv : graph) {
        if (not visited[kv.first]) {
            std::vector<Vertex> component;
            dfs(graph, kv.first, component, additional_restriction);
            components.push_back(component);
            for (auto vertex_id : component) {
                visited[vertex_id] = true;
            }
        }
    }
}

// get the path from the source to the target
template <typename Vertex,
          typename Callable = std::function<bool(Vertex, Vertex)>,
          typename Graph = typename std::enable_if<
              std::is_arithmetic<Vertex>::value, DirectedGraph<Vertex>>::type>
void get_path(
    const Graph &graph, Vertex source, Vertex target, std::vector<Vertex> &path,
    Callable &&additional_restriction = [](Vertex, Vertex) { return true; }) {
    std::vector<Vertex> stack;
    std::vector<Vertex> visited;
    std::unordered_map<Vertex, Vertex> parent;
    stack.push_back(source);
    while (!stack.empty()) {
        auto vertex_id = stack.back();
        stack.pop_back();
        if (std::find(visited.begin(), visited.end(), vertex_id) ==
            visited.end()) {
            visited.push_back(vertex_id);
            if (vertex_id == target) { // check if the target is reached
                break;
            }
            for (auto neighbor_id : graph.at(vertex_id)) {
                if (additional_restriction(vertex_id, neighbor_id)) {
                    parent[neighbor_id] = vertex_id;
                    stack.push_back(neighbor_id);
                }
            }
        }
    }
    path.clear();
    auto vertex_id = target;
    // if the target is not reachable from the source
    if (parent.find(vertex_id) == parent.end()) {
        return;
    }
    while (vertex_id != source) {
        path.push_back(vertex_id);
        vertex_id = parent[vertex_id];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
}
} // namespace GeoRd
#endif // __GRAPH_H__