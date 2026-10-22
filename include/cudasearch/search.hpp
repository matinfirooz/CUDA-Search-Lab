#pragma once
#include "graph.hpp"

namespace cs {
Result search_cpu(const Graph&, int start, int goal, Algorithm, const Options& = {});
// Independent heap-based oracle; all returned distances are final.
std::vector<Cost> reference_dijkstra(const Graph&, int start);
std::vector<Cost> reference_bfs(const Graph&, int start);
Result search_cuda(const Graph&, int start, int goal, Algorithm, const Options& = {});
// For DFS/DLS/IDDFS/IDA*/beam/hill: one CUDA thread per independent query.
// All queries share the CSR graph and options, and have per-goal heuristic arrays.
std::vector<Result> search_cuda_batch(const Graph&,
    const std::vector<std::pair<int,int>>& queries, Algorithm, const Options& = {},
    const std::vector<std::vector<Cost>>& heuristics = {});
bool cuda_available();
std::string cuda_device_name();
} // namespace cs
