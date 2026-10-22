# Benchmark methodology

The repository contains CPU example measurements and source/build validation.
It contains no measured GPU performance or speedup claim.

## What is timed

The CLI measures complete search calls using `std::chrono::steady_clock`,
performs warmups, and reports the median of the requested trials. For an even
trial count it averages the two middle values.

Timed work includes input validation, allocation, graph/heuristic uploads,
kernel execution, host scheduling, synchronization, distance downloads, route
reconstruction, and call teardown. Graph-file parsing, initial heuristic
generation, output serialization, and the optional independent result oracle
are outside the measured region.

For a query file, the scope is the **whole batch**. Each exported query row
repeats that total in `median_end_to_end_ms` and labels `timing_scope=batch`.
The CUDA query-parallel families use one batched kernel; the other families run
the queries sequentially. The CPU reference also runs them sequentially.

Do not sum repeated batch times. Throughput is `batch_size / (batch_ms/1000)`
for one complete batch measurement, not a separate latency for each query.
The C++ batch API likewise returns a total batch interval in each result.
Backend API intervals end before final local buffer teardown; use the CLI
measurement for complete-call comparisons.

## Reproduce

```bash
python3 tools/benchmark.py --exe build/searchlab --backend cuda \
  --sizes 16,32,64 --seed 42 --repeat 7 --warmup 2 --validate \
  --output results/my_gpu
```

The tool stores exact commands, seed, backend, platform, and available
`nvidia-smi` metadata beside the JSON/CSV runs. Grid generation carves a
top-row/right-column route so source and goal are connected. It is a
reproducibility fixture, not an unbiased maze distribution.

Use multiple seeds, obstacle densities, graph sizes, graph families, and query
batch sizes for a performance study. Compare algorithms with the same
objective and guarantee. A faster suboptimal route is a different tradeoff from
a faster exact route. Report preprocessing and end-to-end timing separately if
adding a resident-graph API.

CPU delta-stepping rows execute the independent heap-Dijkstra oracle. The
CPU/CUDA best-first backends also use different priority structures. Do not
describe these comparisons as isolated measurements of identical kernels.

## Example results

`terrain_cpu.json/csv` and `weighted_cpu.json/csv` contain five-trial medians
after one warmup. Their tiny wall times mostly serve as runner examples and
are not suitable evidence of hardware speedups. Paths and expansion counts
are more useful for these fixtures.

Floyd–Warshall reports `expansions=0` because it performs matrix updates rather
than vertex expansions. `iterations=V` is its pivot count. Other expansion
counts reflect their own work units: wave vertices, selected vertices, or
contour/relaxation visits; they are not interchangeable hardware-operation counts.

The desktop GPU run should be checked with Compute Sanitizer before recording
performance. Racecheck primarily diagnoses shared-memory hazards; it is not
a proof of absence of global-memory races. Review the phase/snapshot design
and compare randomized outputs against the independent oracles as well.
