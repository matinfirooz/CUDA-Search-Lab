# Primary references

These references describe the underlying algorithms and CUDA execution rules.
This repository contains an original implementation; it is not a reproduction
of a published high-performance GPU A* system.

- Russell and Norvig, *Artificial Intelligence: A Modern Approach*.
  [Official companion search implementations](https://aimacode.github.io/aima-python/searching.html)
  cover BFS, DFS, DLS, IDDFS, uniform-cost, greedy, A*, IDA*, and local search.
- Hart, Nilsson, Raphael, “A Formal Basis for the Heuristic Determination of
  Minimum Cost Paths,” IEEE TSSC, 1968.
  [DOI](https://doi.org/10.1109/TSSC.1968.300136).
- Dijkstra, “A Note on Two Problems in Connexion with Graphs,” *Numerische
  Mathematik*, 1959.
  [DOI](https://doi.org/10.1007/BF01386390).
- Meyer and Sanders, “Δ-stepping: A Parallelizable Shortest Path Algorithm,”
  *Journal of Algorithms*, 2003.
  [DOI](https://doi.org/10.1016/S0196-6774(03)00076-2).
- NVIDIA, [CUDA C++ Programming Guide, toolkit 12.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html):
  stream ordering, kernel execution, and atomic operations.
- NVIDIA, [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

The best-first priority scans, full-wave bidirectional BFS, dense bucket flags,
and untiled Floyd–Warshall are educational baseline choices. Consult
`IMPLEMENTATION.md` before making performance comparisons.
