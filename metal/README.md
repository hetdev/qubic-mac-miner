# Metal PoC — bpp9000 walk on Apple GPU (experimental)

Clean-room Metal implementation of the mining hot loop (the complete nonce
walk) from the open reference scorer in `../qiner-mac/src/score_bpp9000.h`.
Built to answer one question honestly: **can a Metal trainer mine Qubic on an
Apple GPU?**

## Verdict (measured on Apple M4)

| Aspect | Result |
|---|---|
| Correctness | ✅ **Bit-exact** vs the CPU reference (golden error=4811 + 2 random cases match exactly), for BOTH kernels |
| v1 naive (1 GPU thread = 1 full walk) | ⚠️ 2.1 nonces/s @256 → 8.3 @1024 → 8.0 @4096 (memory-bandwidth plateau) |
| **v2 SIMD (1 simdgroup = 1 walk)** | ✅ 24.7 @256, 27.7 @1024 — 13x one CPU core |
| **v3 (4 walks per threadgroup)** | ✅ **30.8 @1024, 33.4 @4096 — 15x one CPU core, ~5x the 3-thread CPU pool miner** |
| CPU reference | 2.1 nonces/s per core (0.4–0.8 s per walk) |

**Conclusion**: the algorithm ports to Metal perfectly (pure integer math) and
the right SIMT mapping makes the M4 GPU meaningfully faster than the CPU miner.
The only legal intra-nonce parallelism (the node recomputes every nonce with
reference semantics, so the walk trajectory must stay identical) is the 64-neuron
tick step: v2 runs one walk per 32-lane simdgroup with the state in threadgroup
memory and ping-pong neuron buffers.

### Kernel history
- **v1 `bppWalk`** — one thread per walk, device-memory state: validated the
  bit-exactness of the port; memory-bound, ~8 nonces/s ceiling.
- **v2 `bppWalkSimd`** — one simdgroup (32 lanes) per walk; lanes cooperate on
  the 64 neurons per tick; threadgroup-memory state; ~28 nonces/s.
- **v3 `bppWalkSimd4`** — 4 walks per threadgroup (27 KB of the 32 KB TG
  budget), shared accept/rollback/best decisions so all lanes cooperate on
  state copies, simdgroup barriers; **33.4 nonces/s @4096 — 15x one CPU core**.
- Remaining headroom: wider batches, NEON-style preprocessing of seeds,
  GPU-side K12/random2 (removes the CPU staging).
- The result was posted upstream on
  [qubic-li/client#136](https://github.com/qubic-li/client/issues/136#issuecomment-6089967606)
  offering the Metal worker for official integration.

## What the official CUDA trainer actually does

We inspected the pool's worker binaries (`qli-worker-CUDA`, unstripped):

- Real NVIDIA fatbin, targets `sm_61`–`sm_90`.
- Kernel inventory: `kern_solve`, `kern_treeWalk`, `kern_treeWalkScalar`,
  `kern_treeFlip`, `kern_treeEval`, `kern_designGrind`, `kern_generateRandom`,
  `kern_digitTables`, `kern_rootAnn`, `kern_rateParent`,
  `kern_deriveRootMaterial`, `kern_deriveControlOutput`.
- Internal GPU-vs-host fit tests ("the card's fitted error differs from the
  host's…") — the same bit-exactness discipline used here.
- Local state in `current.json` / `.ann` files, sockets on `127.0.0.1`.

**Strategy**: batched candidate evaluation. Instead of walking a whole nonce
per thread, each step generates many candidate mutations/tree moves and
evaluates them in parallel (flip scans, tree evals, design grinds), keeping
the winners tournament-style. That is the redesign a competitive Metal
trainer needs.

## The other wall: the pool protocol

The worker↔client protocol (task distribution + solution submission) is
closed-source. For pool payouts with a Metal worker you must either get
qubic.li to accept/serve one (ask on their Discord / comment on
[qubic-li/client#136](https://github.com/qubic-li/client/issues/136)), or
reverse-engineer the protocol (fragile — breaks on every client update — and
legally gray). Solo mining via the native Qiner port has no such wall.

## Build & run

```bash
./build.sh
./metal_bench validate          # 3 cases, CPU vs GPU bit-exact (v2 kernel)
./metal_bench bench 1024        # batched throughput (v2 SIMD kernel)
```

Files:
- `kernels.metal` — the walk (score/advanceShift/mutate + walk loop), one GPU
  thread per nonce, per-thread state in device memory.
- `metal_bench.mm` — ObjC++ host: derives inputs with the *reference* C++ code
  (K12/random2 stay CPU-side), runs CPU and GPU walks, compares bit-exactly.

## Optimization roadmap (if we pursue this)

1. Batched candidate evaluation instead of full walks (the CUDA strategy):
   kernel = "evaluate K mutations of one network", host/another kernel = keep
   best. Removes the serial walk and the per-step state copies.
2. Per-nonce state in threadgroup memory (currently device memory).
3. SIMD-parallel neuron step (64 neurons across a SIMD group of 32 lanes).
4. GPU-side random2 pool reads + K12 (removes the CPU staging of 80 KB of
   mutation seeds per nonce).
