#ifndef __TEST_GRAPH_H__
#define __TEST_GRAPH_H__

#include "gtest-mpi-listener.h"
#include "../include/GeoRd.h"

using namespace GeoRd;

inline void generate_fully_connected_graph(Graph &g, int n) {
    for (int i = 0; i < n; ++i) {
        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                neighbors.push_back(j);
            }
        }
        add_edge(g, i, neighbors);
    }
}

TEST(Graph, serialization) {
    Graph g;
    generate_fully_connected_graph(g, 4);
    std::vector<std::size_t> ar;
    serialize_graph(g, ar);
    Graph g2;
    deserialize_graph(ar, g2);
    for(const auto &kv : g) {
        ASSERT_EQ(kv.second.size(), g2.at(kv.first).size());
        for(const auto &v : kv.second) {
            ASSERT_TRUE(std::find(g2.at(kv.first).begin(), g2.at(kv.first).end(), v) != g2.at(kv.first).end());
        }
    }
}

inline void generate_undirected_graph(Graph &g) {
    add_edge(g, 0, 1);
    add_edge(g, 1, 0);
    add_edge(g, 0, 2);
    add_edge(g, 2, 0);
    add_edge(g, 1, 3);
    add_edge(g, 3, 1);
    add_edge(g, 2, 3);
    add_edge(g, 3, 2);
    add_edge(g, 4, 5);
    add_edge(g, 5, 4);
}

inline void print_graph(Graph &g) {
    for (const auto &kv : g) {
        std::cout << kv.first << ": ";
        for (const auto &v : kv.second) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
}
TEST(Graph, traverse) {
    Graph g;
    generate_undirected_graph(g);
    std::vector<std::size_t> visited;
    dfs(g, static_cast<std::size_t>(0), visited);
    // Expected [0, 1, 2, 3] are visited
    std::vector<std::size_t> expected = {0, 1, 2, 3};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        auto it = std::find(visited.begin(), visited.end(), expected[i]);
        ASSERT_TRUE(it != visited.end());
    }
    dfs(g, static_cast<std::size_t>(4), visited);
    // Expected [4, 5] are visited
    expected = {4, 5};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        auto it = std::find(visited.begin(), visited.end(), expected[i]);
        ASSERT_TRUE(it != visited.end());
    }
}

TEST(Graph, ConnectedComponents) {
    if(details::MPI_rank() != 0) return;
    Graph g;
    generate_undirected_graph(g);
    std::vector<std::vector<std::size_t>> components;
    connected_components(g, components);
    auto hash_vector = [](const std::vector<std::size_t>& a) {
        std::hash<std::size_t> h;
        std::size_t key = 0;
        for (int i = 0; i < a.size(); ++i) {
            key += h(a[i]) ^ (i * i) << 2;
        }
        return key;
    };
    std::vector<std::size_t> component_key;
    for (const auto &c : components) {
        component_key.push_back(hash_vector(c));
    }
    // Expected 2 components: [[0, 1, 2, 3], [4, 5]]
    std::vector<std::vector<std::size_t>> expected = {{0, 1, 2, 3}, {4, 5}};
    std::vector<std::size_t> expected_key;
    for (const auto &c : expected) {
        expected_key.push_back(hash_vector(c));
    }
    ASSERT_EQ(component_key.size(), expected_key.size());
    for (auto key: component_key) {
        ASSERT_TRUE(std::find(expected_key.begin(), expected_key.end(), key) != expected_key.end());
    }
}

TEST(Graph, GetPath) {
    if(details::MPI_rank() != 0) return;
    Graph g;
    generate_undirected_graph(g);
    std::vector<std::size_t> path;
    get_path(g, static_cast<std::size_t>(0), static_cast<std::size_t>(3), path);
    // Expected path: [0, 1, 3]
    std::vector<std::vector<std::size_t>> possible_path = {{0, 1, 3}, {0, 2, 3}};
    // Check if path is one of the possible path
    bool found = false;
    for (const auto &p : possible_path) {
        if (p.size() == path.size()) {
            bool match = true;
            for (std::size_t i = 0; i < p.size(); ++i) {
                if (p[i] != path[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                found = true;
                break;
            }
        }
    }
    ASSERT_TRUE(found);
}
#endif // __TEST_GRAPH_H__