# CUDA implementation notes

## Two forms of GPU work

Graph-parallel searches upload the CSR graph and coordinate global phases from
the host. BFS expands a complete frontier. Best-first searches select one
minimum-priority vertex on the GPU and relax its row in parallel. Bellman–Ford
and delta-stepping relax many active rows. Floyd–Warshall updates many matrix
cells per pivot.

The DFS family, beam, and hill climbing have serial per-query control flow.
They use the allocation-free bodies in `batched_core.hpp`, compiled for host
and device. A CUDA kernel assigns one thread to each independent query. The
scratch arrays live in global memory, avoiding recursion and a fixed local
stack-size assumption. This layout favors clarity; it is not an optimized
coalesced layout for large batches.

## Phase boundaries and atomics

Kernels use the default stream. A later launch observes completed earlier
work in that stream. Host copies of small control values also synchronize the
needed phase. There is no cross-block spin barrier or assumption that every
block is resident at once.

| Search | Read state | Concurrent write state |
|---|---|---|
| BFS | Current frontier | Atomic distance claims and next-frontier append |
| A*/Dijkstra/greedy | Open-set scan, then frozen selected-source cost | Atomic destination minima, unique per-row parent writes |
| Bellman–Ford | Old distance buffer | Next distance buffer via atomic minima |
| Delta-stepping | Frozen active-source snapshot | Live tentative distances via atomic minima |
| Floyd–Warshall | Frozen pivot row/column | Unique matrix cell per thread |

Distance atomics alone do not synchronize an associated parent field.
Bellman–Ford, delta-stepping, and BFS therefore reconstruct a path on the CPU
from converged **tight edges**. A queue and visited set choose a simple route
even when zero-weight cycles exist. For BFS, tightness uses unit edge cost.

The best-first kernels expand one source at a time. CSR validation enforces
one edge per destination in each row; a successful relaxation therefore has
one writer for that destination's parent in the launch. The selected source
cost is copied into a separate device scalar before relaxation, so source
loads do not race destination atomics on self/back edges.

For Floyd–Warshall, copying the pivot row and column avoids concurrent reads
and writes to pivot cells even though their values would mathematically stay
unchanged for nonnegative weights.

## Delta-stepping

`processed[v]` stores the label at which v last underwent light-edge relaxation.
The next bucket is the minimum `d[v]/delta` among `d[v] < processed[v]`.

Within that bucket:

1. Freeze newly eligible labels; mark those vertices in the removed union.
2. Relax light edges (`w <= delta`).
3. Repeat until no label in the bucket needs reprocessing.
4. Freeze final labels and relax heavy edges from the removed union.

Heavy edges go to later buckets. Improved future labels remain eligible for
later scans. Zero-weight cycles terminate because only strictly lower labels
reactivate vertices.

## Memory and limits

The `--memory-mib` option caps Floyd matrix/pivot storage and serial/batch
scratch. It does not cap all process memory, graph storage, temporary host
copies, or CUDA driver allocations.

A CUDA query needs approximately `56*V` scratch bytes, plus its result and query
IDs: eight int arrays, two cost arrays, and one heuristic array. A 64x64 graph
with 1024 queries needs roughly 224 MiB for this scratch alone. The API checks
the multiplication and budget before allocation.

The expansion budget applies to the best-first family and query-parallel
families. Bulk BFS/SSSP/matrix algorithms run to convergence or their natural
round bound. Depth applies to DLS, IDDFS, and beam; DFS and IDA* use simple-path
limits derived from V.

## Deliberate baseline choices

Priority scans make A* easy to inspect but cause frequent launches and host
round trips. The two BFS waves compute complete fields. The delta scheduler
uses dense flags rather than a sparse bucket queue. Floyd–Warshall is not tiled.
Future performance work can replace these components while preserving the API
and using the independent correctness oracles.
