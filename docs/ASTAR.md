# A* from cost rule to CUDA

A* orders the frontier by `f(v)=g(v)+h(v)`:

- `g(v)` is the best discovered source-to-v cost.
- `h(v)` estimates the remaining cost.
- The goal is accepted when selected as the minimum frontier candidate.

On `examples/weighted.edgelist`, the CLI uses `h=0`, so A* follows Dijkstra.

| Selected vertex | g | Important relaxation |
|---|---:|---|
| 0 | 0 | g(1)=4, g(2)=1, g(5)=50 |
| 2 | 1 | g(1)=3, g(3)=6, g(5)=21 |
| 1 | 3 | g(3)=4, g(4)=10 |
| 3 | 4 | g(4)=7 |
| 4 | 7 | g(5)=8 |
| 5 | 8 | Goal selected |

The returned path is `0→2→1→3→4→5`, cost 8. BFS chooses the one-hop edge
`0→5` with cost 50 because BFS minimizes hops.

In `cuda_search.cu`:

1. `select_key` scans the open vertices and reduces the minimum priority.
2. `select_vertex` picks the lowest vertex ID at that priority.
3. `close_vertex` closes the selected vertex and freezes its g value.
4. `relax_selected` updates successors and reopens strictly improved labels.

These phases run as ordered kernels. The host reads the selected ID to decide
whether to stop or launch another successor expansion.

On a grid, Manhattan distance steers expansion toward the goal while retaining
the optimal-cost guarantee. Weighted A* substitutes `g+w·h`; greedy uses `h`
alone. The implementation shares selection machinery because the difference
is the priority rule, not a different graph representation.

IDA* uses the same `g+h` idea with depth-first contours. Its next threshold
is the smallest f value that exceeded the previous threshold. CUDA runs those
contours independently for each query in a batch.
