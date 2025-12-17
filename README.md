# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository contains Dockerfiles optimized for building and running OpenFold on NVIDIA DGX Spark systems, which feature ARM64 architecture (Grace CPU) and Blackwell GPUs (GB10).

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64, requiring custom builds with specific optimizations to:

1. **Prevent memory exhaustion** - Default builds spawn too many compilation threads
2. **Target correct GPU architecture** - Blackwell GPUs use sm_121 compute capability
3. **Handle ARM-specific dependencies** - Some x86-only optimizations must be disabled

## Quick Start

### OpenFold 2.x

```bash
# Clone OpenFold source first
git clone https://github.com/aqlaboratory/openfold.git
cp openfold2/Dockerfile.spark openfold/
cd openfold
docker build -t openfold2-spark:cuda13 -f Dockerfile.spark .
```

### OpenFold 3.x

```bash
# Clone OpenFold3 source first
git clone https://github.com/aqlaboratory/openfold-3.git
cp openfold3/Dockerfile.spark openfold-3/docker/
cd openfold-3
docker build -t openfold3-spark:cuda13 -f docker/Dockerfile.spark .
```

## Key Optimizations

| Setting | Value | Reason |
|---------|-------|--------|
| MAX_JOBS | 4 | Limits parallel compilation to prevent OOM |
| TORCH_CUDA_ARCH_LIST | 9.0;12.1 | Targets Hopper/Blackwell only |
| DS_BUILD_CPU_ADAM | 0 | Disables x86-only Intel optimizations |
| DS_BUILD_CCL_COMM | 0 | Disables Intel oneCCL (x86-only) |

## System Requirements

- NVIDIA DGX Spark (Grace Blackwell / GB10)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build
- ~60GB RAM during Flash Attention compilation

## License

Apache 2.0 (same as OpenFold)
