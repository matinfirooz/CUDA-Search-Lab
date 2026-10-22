# Contributing

Keep algorithm guarantees explicit. Add a counterexample test when fixing a
search bug, and compare exact routes against independent Dijkstra/BFS oracles.
Do not make a cutoff look like an unreachable result.

CPU:

```bash
cmake -S . -B build-cpu -DCUDASEARCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
python3 tools/check_cli.py --exe build-cpu/searchlab
```

CUDA changes also require a CUDA build, `search_tests --cuda` on an actual GPU,
and Compute Sanitizer. Include hardware/toolkit/driver versions for performance
changes. Label unexecuted GPU tests explicitly.

Use clear C++17, avoid recursion in device search, keep allocations outside
kernels, and document synchronization and buffer ownership.
