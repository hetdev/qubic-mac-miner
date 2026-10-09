# Qubic Mac Miner

Mine [Qubic](https://qubic.org/) on macOS (Apple Silicon) with two engines:

| Phase | Engine | What it is | Payouts |
|---|---|---|---|
| **1** | Official `qli-Client` (qubic.li pool) in Docker | Official pool client + pool worker running x86_64 under **Rosetta** in a lightweight VM (colima, dedicated `qubic` profile) | ✅ PPS (pay per iteration) or solo |
| **2** | **Qiner** (open-source reference miner) ported to macOS/ARM64 | Fork of [qubic/Qiner](https://github.com/qubic/Qiner) that compiles and runs **natively** on Apple Silicon | Solo against a node (lottery-level odds on a Mac) |

> **Why does Phase 1 use Rosetta instead of the arm64 binary?** The qubic.li
> pool pushes an **Android** worker (`/system/bin/linker64`, bionic libc) to
> Linux ARM64 clients — it can never start on glibc Linux. Open upstream bug:
> [qubic-li/client#136](https://github.com/qubic-li/client/issues/136).
> The x64 worker *is* a proper glibc Linux build, so the stack runs x86_64
> translated by Rosetta (~70–90 % of native). When #136 is fixed, switch back
> to `QLI_ARCH=arm64` + `platform: linux/arm64`.

```
qubic-mac-miner/
├── docker-compose.yml        # Phase 1: pool services (per-pool profiles)
├── docker/Dockerfile         #   x64 image of qli-Client 3.9.1 (via Rosetta)
├── .env.example              #   your settings (public address, NEVER the seed)
├── scripts/qubicminer        # CLI: start/stop/status/logs/update/bench
├── launchd/                  # auto-start at login
├── docs/EXPECTATIONS.md      # measured performance, validation, risks
└── qiner-mac/                # Phase 2: Qiner fork with the ARM64 port
```

---

## Phase 1 — Mine with a pool TODAY (recommended)

### 1. Prerequisites (one time)

1. **Wallet**: create your address at <https://wallet.qubic.org> and keep the
   seed **offline** (paper/1Password). The seed is **never** used in this project.
2. **Pool**: create an account at <https://pool.qubic.li> /
   <https://platform.qubic.li> to see your dashboard; registerless mode only
   needs your public address.
3. **Docker runtime** (pick one):
   - **colima** (recommended, lightweight): `brew install colima docker docker-compose`
     — the `qubicminer` CLI creates and starts the `qubic` profile by itself
     (`--vm-type vz --vz-rosetta`, 4 CPU / 6 GB) without touching your other VMs.
   - Docker Desktop (GUI, free license for personal use) — in that case export
     `QUBIC_DOCKER_CONTEXT=default` before using the CLI.

### 2. Configuration

```bash
cp .env.example .env
# edit .env: QUBIC_ADDRESS (your public address) and optionally QUBIC_TOKEN
# threads: defaults to (P-cores - 1); on a base M1/M2/M3 that is 3
```

### Supported pools (`QUBIC_POOL` in `.env`)

| Pool | Value | Mode | Status on Mac |
|---|---|---|---|
| **QLI / [platform.qubic.li](https://platform.qubic.li/platform)** (primary) | `qli` | PPS or solo | ✅ Verified (via Rosetta, ~19 it/s on M-series) |
| [jetskipool.ai](https://qubic.jetskipool.ai/) | `jetski` | PPLNS or solo | ⏳ Intel Macs only today; on Apple Silicon their AVX2 trainer dies with SIGBUS under Rosetta and they publish no arm64 build yet ("Apple Silicon coming soon" on their site). Integration is ready for when they do |
| Any **QLI-protocol-compatible** endpoint | `qli` + `QUBIC_POOL_URL=wss://…` | PPS or solo | Same as QLI |

The CLI starts only the selected pool (compose profiles); both share the
`qubic-miner` container name, so `status`/`logs`/`stop` work the same. On
jetski, `QUBIC_PPS=true` maps to `--pplns` and `false` to `--solo`; no token is
used (address only).

### 3. Start

```bash
./scripts/qubicminer start     # starts the pool (first run: builds the image)
./scripts/qubicminer status    # container state, it/s and resource usage
./scripts/qubicminer logs      # live logs
./scripts/qubicminer stop      # clean shutdown
```

Check your worker at <https://pool.qubic.li> (log in with your address).

**Note**: the client may log `Auth: no AuthenticateResponse from pool after 3
attempts` — it shows up in registerless mode **and with a token** (a client
quirk, not a config problem); the pool assigns work anyway and payouts are
credited to your address. On a Mac the share counter moves slowly (low
hashrate); use `it/s` from `status` to confirm activity.

### 4. Auto-start at login

```bash
./scripts/qubicminer install-service    # LaunchAgent (starts at login)
./scripts/qubicminer uninstall-service  # remove
```

With `MINER_ON_BATTERY=skip` in `.env`, the service will not start on battery.

---

## Phase 2 — Native Qiner (ARM64 port, engineering project)

The official reference miner is C++ for Windows/Linux x86. This fork
(`qiner-mac/`) **compiles and runs natively on Apple Silicon** with minimal,
validated changes (see `docs/EXPECTATIONS.md` for the validation detail):

| Change | Why |
|---|---|
| `src/portable_rng.h` | Shims for the x86 `_rdrand*_step` intrinsics (uses `arc4random_buf`, the system CSPRNG) |
| `src/K12AndKeyUtil.h` | Portable FourQ arithmetic: `_addcarry_u64`/`_subborrow_u64` via `__int128`, 128-bit shifts without x86 asm, and a 32-byte vector type instead of `__m256i` |
| `CMakeLists.txt` | `-mrdrnd -mbmi` flags only on x86 |
| `src/Qiner.cpp` | **Shared random2 pool**: 1 × 512 MB for all threads (was: 512 MB *per thread*) |

Port patches are also exported as `qiner-mac/patches/*.patch` (upstream base:
`qubic/Qiner@71cfbd6`), so the delta is reviewable without git history.

### Build and validate

```bash
cd qiner-mac
./build-macos.sh arm64      # or: x86_64 | universal
./build-macos/qiner_test    # validation suite (digests + shared pool)
```

(With cmake: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.)

### Run in solo mode (needs a node + real task)

```bash
./build-macos/Qiner <Node IP> 31841 <MiningID> <Signing Seed> <Mining Seed hex> <threads> <task_bpp9000.bin>
```

- The real `task_bpp9000.bin` is served by the epoch's node; local benchmarks
  use the canonical task `test/data/bpp9000.task` (upstream's
  `data/example_task_bpp9000.bin` is pre-trimode and the loader rejects it).
- Seeds/identities: see the upstream README. The official README's example
  identity (`BZBQ…QEXK`) derives from seed `aaa…a` — we reproduce it
  bit-exact on ARM64 (part of the port validation).

### Local benchmark (no node, submits nothing)

```bash
./scripts/qubicminer bench 4 300   # 4 threads, 5 min
```

---

## Security

- **Never** put your wallet seed in `.env`, in the pool or in the Qiner build of
  this repo. Only the **public address** (or the pool's JWT token).
- The pool's trainer is a **closed-source binary** the client downloads: it runs
  with least privilege, inside the Docker VM, with no access to your home.
- Rosetta 2 is used in **Phase 1** (it translates the pool's x64 worker because
  of upstream bug #136) and for the cross-arch validation of the Phase 2 port;
  the native Qiner does not need it.

## Troubleshooting

**`brew install docker` fails with "directories are not writable"**
Your Homebrew has directories owned by another user. The CLI still lands in the
Cellar: the `qubicminer` script creates a `bin/docker` symlink automatically.
To fix brew properly: `sudo chown -R $(whoami) /opt/homebrew` (not needed for
this project).

**`docker compose` is missing**
No problem: the CLI falls back to standalone `docker-compose`
(`brew install docker-compose`).

**colima does not start on its own**
`./scripts/qubicminer start` tries it; manually: `colima start --profile qubic`.
If the docker CLI is not on the global PATH, start colima like this:
`PATH="$PWD/bin:$PATH" colima start --profile qubic`.

**The container never becomes healthy**
`./scripts/qubicminer logs` — with an invalid address in `.env` the client
fails to authenticate against the pool (expected).

## Licenses

- `qiner-mac/`: fork of `qubic/Qiner` — original "Anti-Military" license in
  `LICENCE` + `NOTICE.md` (kept intact).
- qli-Client 3.9.1: official binary redistributed unmodified from
  <https://dl.qubic.li> (qubic.li terms).
- Everything else in this repository: MIT (see `LICENSE`).

## More

- Measured performance and risks: [`docs/EXPECTATIONS.md`](docs/EXPECTATIONS.md)
- Qubic ecosystem: <https://qubic.org> · docs: <https://docs.qubic.org> ·
  pool: <https://pool.qubic.li>
