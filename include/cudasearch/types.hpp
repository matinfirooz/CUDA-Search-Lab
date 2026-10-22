#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cs {
// unsigned long long is intentional: it matches CUDA's 64-bit atomicMin overload.
using Cost = unsigned long long;
constexpr Cost INF = (1ULL << 60);
enum class Algorithm {
    Bfs, BidirectionalBfs, Dfs, DepthLimited, IterativeDeepening,
    Dijkstra, AStar, WeightedAStar, Greedy, IdaStar, Beam, HillClimbing,
    BellmanFord, DeltaStepping, FloydWarshall
};
enum class Status { Found, Unreachable, LimitReached, LocalMinimum };
struct Options {
    int depth_limit = 64;
    int beam_width = 32;
    Cost heuristic_weight = 2;
    Cost delta = 4;
    Cost max_expansions = 1000000;
    std::size_t memory_limit_bytes = 512ULL * 1024 * 1024;
    std::vector<Cost> heuristic; // empty means h=0
};
struct Result {
    Status status = Status::Unreachable;
    Cost cost = INF;             // sum of original edge weights
    Cost objective = INF;        // hops for BFS/bidirectional BFS/IDDFS, cost otherwise
    std::vector<int> path;
    Cost expansions = 0;
    Cost iterations = 0;
    double elapsed_ms = 0;      // backend wall interval; CLI times complete calls
};
Algorithm parse_algorithm(const std::string&);
std::string name(Algorithm);
std::string name(Status);
std::vector<Algorithm> algorithms();
bool is_batched(Algorithm);
bool is_hop_optimal(Algorithm);
bool is_cost_optimal(Algorithm);
inline Cost add(Cost a, Cost b) { return a >= INF || b >= INF || a > INF-b ? INF : a+b; }
} // namespace cs
