# Algorithms and guarantees

All guarantees assume a finite valid graph with nonnegative weights and enough
time, memory, and expansion budget. “Optimal hops” and “optimal cost” differ on
weighted graphs.

| Command | Ordering / rule | Guarantee on a found route | Limitations |
|---|---|---|---|
| bfs | Complete FIFO distance levels | Minimum hops | Original weights are ignored during search |
| bidirectional-bfs | Forward and transpose BFS, minimum-hop meeting | Minimum hops | Two full waves; no early-stop optimization |
| dfs | CSR successor order, graph visited set | Valid route | No cost or hop bound |
| dls | Depth-first simple paths within depth | Valid route within depth | Exponential; a depth cutoff is not unreachable |
| iddfs | DLS at depths 0,1,… | Minimum hops | Exponential; maximum depth/budget can cut off |
| dijkstra / ucs | Minimum tentative cost | Minimum weighted cost | CUDA uses dense priority scans |
| astar | Minimum g+h | Minimum weighted cost | Custom heuristic must be consistent and zero at goal |
| weighted-astar | Minimum g+w·h; reopen improved labels | Cost at most w times optimum | Integer w≥1, consistent h; exactness not promised |
| greedy | Minimum h; reopen improved labels | Valid route | No quality bound; h=0 gives vertex-ID tie ordering |
| idastar | Depth-first contours bounded by g+h | Minimum weighted cost | Simple-path cycle checks; exponential |
| beam | Layer top-W by g+h | Valid route within depth | Incomplete; selected vertices are not reopened |
| hill | Neighbor with strictly lower h | Valid route if goal reached | Plateaus and local minima can stop search |
| bellman-ford | Double-buffered relaxation, ≤V−1 rounds | Minimum weighted cost | This common graph format rejects negative weights |
| delta-stepping | Minimum bucket, light closure, heavy edges | Minimum weighted cost | Baseline scans V labels for bucket selection |
| floyd-warshall | All-pairs pivot recurrence | Minimum weighted cost | O(V³) work, O(V²) storage, intended for small graphs |

The `ucs` command is an alias for Dijkstra, not a sixteenth distinct algorithm.
The release contains 15 unique implementations.

## Status values

- `found`: a valid route was returned; apply the guarantee in the table.
- `unreachable`: the search exhausted the relevant finite state space without
  pruning that could hide a route.
- `limit-reached`: a depth, expansion, or beam pruning limit prevents a conclusion.
- `local-minimum`: hill climbing has no strictly improving neighbor.

A beam search that discarded candidates and then runs out of retained candidates
reports `limit-reached`. A bounded depth search uses on-path cycle detection,
rather than a global visited set, so a vertex first reached too deeply can still
be explored through a shorter prefix.

## Complexity and parallelism

CSR storage is O(V+E). BFS is O(V+E) work plus host/device frontier orchestration.
The CUDA best-first family scans V labels for each expanded vertex and relaxes
that vertex's successors in parallel. Dijkstra therefore has O(V²+E) scan work,
not the O((V+E) log V) work of a heap implementation. Weighted A* and greedy can
reopen labels, so use their actual expansion count K when estimating O(KV) scans.

DFS is O(V+E) with a global visited set. DLS, IDDFS, and IDA* can enumerate
exponentially many simple paths. The serial bodies use explicit O(V) scratch,
and CUDA runs many query bodies concurrently. Beam scratch is O(V) per query,
with linear scans to select each retained candidate.

Bellman–Ford performs O(VE) edge work in the worst case. Delta-stepping work
depends on bucket width and light-edge reactivations; this implementation also
scans O(V) state per bucket/closure. Floyd–Warshall performs O(V³) cell updates.

## Heuristics and numeric domain

For custom A*/IDA*/weighted-A* heuristics, every edge must satisfy
`h(u) <= w(u,v) + h(v)` and `h(goal)=0`. These conditions imply admissibility
for vertices that can reach the goal. The implementation deliberately checks
consistency instead of silently accepting arbitrary heuristic claims.

Manhattan distance is consistent for the generated four-neighbor grids because
every edge costs at least 1. General graph CLI inputs use zero.

Costs are uint64; edge weights are uint32. `INF = 2^60` is reserved. Validation
rejects graphs whose conservative simple-path bound reaches `INF/4`, oversized
heuristics, and weighted priorities that reach the sentinel. Input vertex/edge
counts must fit signed 32-bit CSR offsets.
