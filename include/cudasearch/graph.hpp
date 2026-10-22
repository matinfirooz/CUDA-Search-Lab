#pragma once
#include "types.hpp"
#include <istream>
#include <tuple>

namespace cs {
struct Graph {
    int n = 0;
    std::vector<int> row, col;
    std::vector<unsigned int> weight;
    static Graph from_edges(int n, std::vector<std::tuple<int,int,Cost>> edges);
    static Graph read(std::istream&);
    Graph transpose() const;
    void validate() const;
    Cost edge_cost(int u, int v) const;
};
struct Grid {
    Graph graph;
    int width = 0, height = 0, start = -1, goal = -1;
    std::vector<std::string> cells;
    std::vector<Cost> manhattan(int target) const;
    static Grid read(std::istream&);
    static Grid random(int width, int height, double obstacles, unsigned seed);
};
void validate_request(const Graph&, int start, int goal, Algorithm, const Options&);
Cost path_cost(const Graph&, const std::vector<int>&);
// Reconstruct on tight edges after distances stabilize. This avoids incoherent
// (distance,parent) pairs and parent cycles on zero-weight edges.
std::vector<int> tight_path(const Graph&, int start, int goal,
                            const std::vector<Cost>& distance, bool hops = false);
Result result_from_distances(const Graph&, int start, int goal,
                             const std::vector<Cost>& distance, bool hops = false);
} // namespace cs
