Here is the comprehensive guide to deploying OpenFold on the **NVIDIA DGX Spark** (Grace Blackwell / GB10) with CUDA 13\.

The crash you are experiencing is caused by **Unified Memory Exhaustion**. The build system (ninja) spawns threads for all 72+ CPU cores. Compiling CUDA 13 kernels requires \~4GB RAM per thread ($72 \\times 4\\text{GB} \\approx 288\\text{GB}$). This demand instantly exceeds the system's 128GB Unified Memory, causing the OS to hang.

### **1\. Host System Prerequisites**

Before Dockerizing, verify the host is capable of running Blackwell code.

* **Verify Driver:** Run nvidia-smi. The driver must be **580.xx** or higher. Older drivers cannot execute binaries for the sm\_121 (GB10) architecture.  
* **Verify Architecture:** Run the following to confirm the GPU Compute Capability (typically 12.1 for GB10):  
* Bash

```

python3 -c "import torch; print(torch.cuda.get_device_capability(0))"

```

* 

---

### **2\. The Dockerfile**

Save the following as Dockerfile.spark.

**Key Technical Interventions:**

* **MAX\_JOBS=4**: Throttles the build to 4 threads, preventing the memory crash.  
* **sm\_121**: Targets the specific Blackwell architecture of the DGX Spark.  
* **\--no-build-isolation**: Forces the build to use our custom PyTorch Nightly (CUDA 13\) instead of downloading an incompatible stable version.

Dockerfile

```

# 1. Base Image: NVIDIA ARM64 Optimized
# Use the latest available tag (e.g., 25.11 or 25.12) to ensure CUDA 13 toolkit components are present.
FROM nvcr.io/nvidia/pytorch:25.12-py3

# ---------------------------------------------------------------------------
# CRITICAL FIX: PREVENT SYSTEM HANG (Unified Memory Protection)
# The Grace CPU has 72+ cores. Ninja defaults to 1 thread per core.
# We limit to 4 concurrent jobs to stay within the 128GB memory limit.
# ---------------------------------------------------------------------------
ENV MAX_JOBS=4
ENV NVCC_THREADS=4
ENV OMP_NUM_THREADS=4

# 2. Target Architecture: Blackwell (GB10)
# GB10 identifies as sm_121. We also include 9.0 (Hopper) as a fallback.
ENV TORCH_CUDA_ARCH_LIST="9.0;12.1"
ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"

# 3. System Dependencies
# libaio-dev is strictly required for DeepSpeed on ARM64.
RUN apt-get update && apt-get install -y \
    wget git ninja-build libxml2 libaio-dev \
    hmmer kalign hhsuite \
    && rm -rf /var/lib/apt/lists/*

# 4. PyTorch Nightly (CUDA 13.0)
# We uninstall the container's default Torch to force-install the Nightly build
# that explicitly supports the sm_121 Blackwell architecture.
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --no-cache-dir

# 5. Biology Tools via Conda (The "Easy Way" for ARM64)
# OpenMM is notoriously difficult to compile on ARM. We use Miniforge to
# fetch pre-compiled binaries from the conda-forge channel.
ENV CONDA_DIR=/opt/conda
RUN wget -qO ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Install OpenMM and PDBFixer
RUN mamba install -y -c conda-forge openmm pdbfixer && mamba clean -afy

# 6. FlashAttention (The "Crash Zone")
# This step will take 20-30 minutes due to MAX_JOBS=4.
# --no-build-isolation is VITAL so it uses the Nightly PyTorch headers.
RUN pip install packaging && \
    pip install flash-attn --no-build-isolation --no-cache-dir

# 7. DeepSpeed
# We force the build of ops (DS_BUILD_OPS=1) to ensure the JIT compiler
# targets the Neoverse V2 CPU correctly.
ENV DS_BUILD_OPS=1
RUN pip install deepspeed --no-build-isolation

# 8. OpenFold Installation
WORKDIR /opt
RUN git clone https://github.com/aqlaboratory/openfold.git
WORKDIR /opt/openfold

# Patch requirements.txt: Remove lines that would downgrade our custom builds.
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/flash-attn/d' requirements.txt && \
    sed -i '/deepspeed/d' requirements.txt

# Install dependencies and OpenFold
RUN pip install biopython ml_collections scipy dm-tree pytorch_lightning wandb modelcif
RUN python3 setup.py install

# 9. Runtime Validation
CMD ["python3", "-c", "import torch; print(f'CUDA: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}'); import openfold; print('OpenFold Installed Successfully')"]

```

---

### **3\. Build & Run Instructions**

#### **Step 1: Build the Image**

Be patient. The build will appear to "hang" during the flash-attn step. **Do not cancel it.** Check htop on the host; you will see 4 nvcc processes working hard.

Bash

```

docker build -t openfold-spark:cuda13 -f Dockerfile.spark .

```

#### **Step 2: Run the Container**

You must configure the runtime environment to handle OpenFold's heavy dataloading and the Unified Memory architecture.

Bash

```

docker run --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/data:/data \
    -it openfold-spark:cuda13

```

*   
  \--ipc=host: Essential for PyTorch dataloaders in shared memory.  
* \--ulimit stack=...: Increases stack size to prevent segmentation faults during complex protein recursions.

---

### **4\. Component Compatibility Guide (How to Find Versions)**

Since this hardware is bleeding edge, standard compatibility charts often fail. Here is how to verify each component manually:

| Component | Strategy | How to Find / Verify |
| :---- | :---- | :---- |
| **PyTorch** | **Nightly (cu130)** | Standard releases likely lag behind sm\_121 support. Go to the [PyTorch Nightly Index](https://www.google.com/search?q=https://download.pytorch.org/whl/nightly/cu130/) and look for wheels ending in cp311-linux\_aarch64.whl. Ensure the URL explicitly contains cu130 (CUDA 13.0). |
| **FlashAttention** | **Source Build** | Do NOT use pip install flash-attn (it downloads x86 wheels). You must build from source. Monitor the [FlashAttention Releases](https://github.com/Dao-AILab/flash-attention/releases); if you do not see aarch64 wheels, assume source build is required. |
| **OpenMM** | **Conda-forge** | Never build from source on ARM unless necessary. Go to [Anaconda.org/conda-forge/openmm](https://anaconda.org/conda-forge/openmm) and filter files by linux-aarch64. If a recent version exists, use mamba install. |
| **DeepSpeed** | **Source w/ Flags** | DeepSpeed's JIT compiler often hangs on Grace CPUs. Force build-time compilation with DS\_BUILD\_OPS=1. If it hangs at *runtime*, set DS\_BUILD\_OPS=0 to fall back to pure Python. |
| **Target Arch** | **sm\_121** | If you get "no kernel image is available" errors, your build targeted the wrong architecture. Always verify your card's capability using the python one-liner in Section 1 and ensure TORCH\_CUDA\_ARCH\_LIST matches it. |

