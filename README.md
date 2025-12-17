# OpenFold for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)

This repository contains Dockerfiles optimized for building and running OpenFold on NVIDIA DGX Spark systems.

> **Looking for OpenFold3?** See [openfold3-DGX-Spark](https://github.com/adriancarr/openfold3-DGX-Spark)

## Why This Exists

The official OpenFold Docker images are built for x86_64 architecture. DGX Spark uses ARM64, requiring custom builds with specific optimizations.

## Quick Start

```bash
# Clone OpenFold source
git clone https://github.com/aqlaboratory/openfold.git
cd openfold

# Copy the DGX Spark Dockerfile
curl -O https://raw.githubusercontent.com/adriancarr/openfold-DGX-Spark/main/openfold/Dockerfile.spark

# Build (takes ~45 minutes due to Flash Attention compilation)
docker build -t openfold-spark:cuda13 -f Dockerfile.spark .

# Test
docker run --gpus all --rm openfold-spark:cuda13 python3 -c \
  "import openfold; print('OpenFold ready')"
```

## Key Optimizations

| Setting | Value | Reason |
|---------|-------|--------|
| `MAX_JOBS` | 4 | Limits parallel compilation to prevent OOM |
| `TORCH_CUDA_ARCH_LIST` | `9.0;12.1` | Targets Hopper/Blackwell only |
| `DS_BUILD_CPU_ADAM` | 0 | Disables x86-only Intel optimizations |
| `DS_BUILD_CCL_COMM` | 0 | Disables Intel oneCCL (x86-only) |

## System Requirements

- NVIDIA DGX Spark (Grace Blackwell / GB10)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for build
- ~60GB RAM during Flash Attention compilation

## Notes

- PyTorch Nightly with CUDA 13.0 is used for Blackwell GPU support
- A warning about compute capability 12.1 may appear - this is expected and safe to ignore

## Credits

- [OpenFold](https://github.com/aqlaboratory/openfold) by AlQuraishi Lab

## License

Apache 2.0 (same as OpenFold)
