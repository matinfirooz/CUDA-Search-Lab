# Recorded validation

Validated on 2026-10-22. This record separates host checks, compilation, and
GPU execution.

| Check | Observed result |
|---|---|
| GNU C++17 release build | Passed, GCC 13.3.0 |
| C++ CPU correctness suite | 4,921 search cases; 34,087 assertions passed |
| CPU CTest suite | 3/3 passed |
| CPU AddressSanitizer + UBSan | 3/3 CTest checks passed |
| JSON/CSV CLI integration | Passed; 15 algorithm rows and expected weighted costs |
| Invalid CLI arguments | 9 rejection cases passed |
| Mixed-goal query file | CPU reference validation passed |
| Benchmark runner | 4x4 and 8x8 BFS/A*/Dijkstra-reference delta rows passed validation |
| CUDA source compilation | Passed, nvcc 13.4.92, C++17, target SM 75 |
| CUDA-linked CMake build | Both executables linked successfully |
| Host checks in CUDA-linked build | 3 passed |
| CUDA runtime correctness test | **Skipped: no CUDA GPU in this workspace** |
| CUDA Compute Sanitizer | **Not run: no GPU** |
| SVG comparison | Rendered and visually inspected |
| Offline HTML viewer | JavaScript syntax checked; browser interactions not executed |
| GitHub Actions | Workflow supplied; not run on GitHub here |

LeakSanitizer could not inspect this workspace's process/thread filesystem.
The CPU sanitizer run used `ASAN_OPTIONS=detect_leaks=0`, retaining address
and undefined-behavior checking. This is not a leak-detection result.

## Reproduce the host checks

```bash
cmake -S . -B build-cpu -DCUDASEARCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
./build-cpu/search_tests
python3 tools/check_cli.py --exe build-cpu/searchlab

cmake -S . -B build-sanitize -DCUDASEARCH_ENABLE_CUDA=OFF \
  -DCUDASEARCH_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure
```

Only add `ASAN_OPTIONS=detect_leaks=0` to the final command if your own
environment prevents LeakSanitizer from operating.

## GPU checks to run on hardware

```bash
cmake -S . -B build -DCUDASEARCH_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/search_tests --cuda
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/search_tests --cuda
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/search_tests --cuda
compute-sanitizer --tool initcheck --error-exitcode 1 ./build/search_tests --cuda
```

The GPU suite checks fixed and randomized graphs and 257 mixed-goal queries per
batched search family. It is implemented but was not executed in this workspace.
Compilation does not establish the absence of device runtime or concurrency bugs.

The bundled CPU example files are runner demonstrations. They do not provide
evidence of CUDA speedup.
