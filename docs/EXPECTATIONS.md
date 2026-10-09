# Expectations, Validation and Risks

Read this before leaving your Mac mining. Honest summary: **a Mac is not a
mining rig**; this is a technically correct, working project, not a money
machine.

## Expected performance

- The current algorithm (**bpp9000**) is a search over a recurrent ternary
  network: each nonce simulates the network with up to 1,000 mutations and
  rollouts of a 8,760-emission window.
- **Measured on this Mac (M-series, 4P+6E cores)**: a full nonce walk takes
  **0.4–0.8 s per core**; with 4 threads that is ~**7 it/s** sustained
  (2,156 nonces in 280 s ≈ 1.9 nonces/s per thread).
- The **pool** (Phase 1) is the only route with realistic payouts: in **PPS**
  mode you are paid per iteration. **Measured on this Mac: ~19 it/s** with 3
  threads (GENERIC x64 worker under Rosetta in the VM; ≈70–90 % of native x64).
  On an M-series expect **cents or fractions of a cent per day**, not more
  (AVX-512 x86 rigs with dozens of threads dominate the hashrate).
- **Why Rosetta**: the pool pushes an Android (bionic) worker to Linux ARM64
  clients that cannot start on glibc — upstream bug
  [qubic-li/client#136](https://github.com/qubic-li/client/issues/136), open
  with no fix since August 2026. The x64 worker is correct, so the stack runs
  translated x86_64. When it is fixed, go back to native arm64.
- **Solo** (Phase 2 against a node): finding a solution that wins a computor
  slot with a Mac is essentially a lottery. The value of the port is the
  engineering: a 100 % native, validated reference miner.
- Note: upstream's `data/example_task_bpp9000.bin` uses pre-trimode dimensions
  and the current loader **rejects** it; always use the canonical task
  `test/data/bpp9000.task` (the `bench` CLI already does).

## Power and thermals

- 3–4 threads at full load = fans at maximum on laptops; expect throttling.
- Recommendation: `QUBIC_THREADS = P-cores - 1` and the Mac on AC power.
- `MINER_ON_BATTERY=skip` in `.env` avoids mining on battery.

## Phase 2 ARM64 port validation — what was proven

1. **`deriveRootANN` root digests**: the 3 committed upstream golden vectors
   produce an exact MATCH on ARM64.
2. **Cross-arch (Rosetta)**: the same code compiled for x86_64 and run under
   Rosetta 2 produces results identical to the arm64 binary (scores, digests
   and FourQ identities byte for byte).
3. **Public known vector**: seed `aaa…a` (55×'a') derives identity
   `BZBQFLLBNCXEMGLOBHUVFTLUPLVCPQUASSILFABOFFBCADQSSUPNWLZBQEXK` — exactly the
   example identity in the official Qiner README.
4. **Shared pool**: owned-pool vs shared-pool equivalence test (`setPool`) in
   `qiner_test` — identical Ratings; flat memory (~512 MB + stacks, not
   N×512 MB).
5. **Suite**: `./build-macos/qiner_test` green (17 assertions).

### Note on upstream's golden CSVs

`test/data/gt_production.csv` and `gt_ant_production.csv` were captured
**before** the "trimode" merge changed the walk parameters and the ant-colony
API: with the current code (x86 **and** ARM64, identical to each other) row 0
scores 4881 vs the CSV's 4254. They are disabled behind
`QINER_ENABLE_STALE_CSV_GOLDENS` (they fail on every platform equally; regenerate
with `tools/bpp9000_ground_truth.cpp`). Upstream has no CI to catch this — worth
filing as an issue.

## Risks

- **Closed-source pool worker**: downloaded by qubic.li; you run their code in
  a VM with least privilege. If you do not trust it, use Phase 2 only
  (100 % buildable-from-source code).
- **Algorithm changes per epoch**: Qubic has changed the algorithm 4+ times.
  The pool absorbs that for you; standalone Qiner requires updating the fork.
- **amd64 Docker image + Rosetta**: the official `qubicli/client` image is
  amd64-only and the pool's arm64 worker is Android (bug #136); that is why we
  use the **official x64 tarball** in our own image running translated. Rosetta
  loses ~10–30 % of performance; that is the price of a worker that works at
  all. On Intel Macs use `QLI_ARCH=x64` without Rosetta (native).

## Verification performed on this Mac (Apple Silicon, macOS 27)

1. **Full integration against the qubic.li pool** (colima profile `qubic`,
   VZ+Rosetta): WSS connection ✓, task + Mining Identity assigned ✓, GENERIC
   worker downloaded and running ✓, epoch 234 trainer at ~19 it/s ✓, container
   `healthy` ✓, clean shutdown ✓. **Confirmed on the pool dashboard** (worker
   `mac-mini` Active, 18 H/s, score 3 — see
   [`docs/assets/pool-worker-active.webp`](assets/pool-worker-active.webp)).
2. `qiner-mac` compiled natively arm64 (Qiner + AntMiner + tests) with a green
   test suite.
3. The cross-arch bit-exact validations listed above.
4. Native benchmark with the canonical task: ~7 it/s with 4 threads
   (2,156 nonces in 280 s), clean SIGINT shutdown.

### Findings from the review (fixed in this project)

- **Upstream bug #136**: the pool pushes an Android worker to Linux ARM64 →
  x64+Rosetta workaround documented above.
- The original healthcheck used `ps`, which does not exist in
  `debian:bookworm-slim` (always `unhealthy`) → now inspects `/proc/1/cmdline`.
- launchd runs with a minimal PATH: the CLI now includes `/opt/homebrew/bin`
  (colima, docker-compose) plus its own `bin/`.
- The classic builder of standalone `docker-compose` fails with Docker 29's
  image store → the image is built with `docker build` (done by the CLI).
- `Auth: no AuthenticateResponse…` appears in registerless mode **and with a
  token** (client quirk); documented, not a config problem.
- jetskipool.ai was tested: their client connects and receives configuration,
  but their AVX2 trainer dies with SIGBUS under Rosetta and `SKYLAKE` is gated
  by a CPUID signature check; their site announces Apple Silicon "coming soon".
  Works natively on Intel Macs. Integration (`QUBIC_POOL=jetski`) is ready.

## Metal / Apple GPU (experimental PoC)

`metal/` contains a clean-room Metal implementation of the bpp9000 nonce walk,
validated **bit-exact** against the CPU reference (golden 4811 included).
Measured on Apple M4: the naive "one thread per walk" pattern plateaus at
~8 nonces/s, but the SIMT mapping that preserves reference semantics (one
32-lane simdgroup per walk, 64-neuron tick step across lanes, threadgroup-
memory state) reaches **~28 nonces/s ≈ 13x one CPU core and ~4.3x the 3-thread
CPU pool miner**. Details, kernel history and roadmap:
[`metal/README.md`](../metal/README.md).

## What this project is NOT

- Not a port of the pool client to a native macOS binary (the trainer is
  closed source; impossible from source).
- No GPU/Metal (the pool publishes no Metal trainer; the Qiner bpp9000
  algorithm is CPU).
- No profitability promise. If you want serious Qubic hashrate, the right
  hardware is x86 with AVX-512 and lots of memory channels.
