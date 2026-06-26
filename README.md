# MCTLASS/CUTLASS GEMM Demo

This repository contains a small self-checking GEMM demo for MetaX MCTLASS
and NVIDIA CUTLASS.

The executable computes:

```text
D = alpha * A * B + beta * C
```

with row-major matrices, then compares the device result against a CPU
reference implementation. The MetaX/MCTLASS backend uses FP32 with `beta=0`
through a direct MCTLASS grouped GEMM kernel. The CUDA/CUTLASS backend uses
FP32 with `beta=1`.

## Requirements

- GNU Make
- For MetaX C500: MACA SDK with `cucc` and MCTLASS headers
- For NVIDIA: CUDA toolkit with `nvcc` and CUTLASS headers

On this machine the Makefile defaults to `BACKEND=maca` when it finds
`/opt/maca/include/mctlass/mctlass.h`.

## Build

MetaX/MACA + MCTLASS:

```bash
make BACKEND=maca ARCH=80
```

NVIDIA CUDA + CUTLASS:

```bash
make BACKEND=cuda CUTLASS_DIR=/path/to/cutlass ARCH=80
```

Set `ARCH` for the CUDA-compatible architecture accepted by your compiler, for
example `70`, `75`, `80`, `86`, `89`, or `90`.

## Run

```bash
./build/maca_gemm_demo
./build/maca_gemm_demo 512 512 512
./build/maca_gemm_demo --mcpti 4096 4096 4096
```

The optional arguments are `M N K`. The MCTLASS FP32 kernel requires `K` to be
a multiple of 4. `--mcpti` enables MetaX MCPTI counter collection for DRAM and
L2 transaction traffic on the MCTLASS backend.

The program also prints estimated GEMM matrix traffic in Bytes for HBM and for
the L1-L2 interface. HBM traffic is the logical tensor read/write traffic. L1-L2
traffic is estimated from the backend tile shape used by the demo.

You can also build and run the default problem size with:

```bash
make run BACKEND=maca ARCH=80
```

## Gap 1 Benchmarks

The Orojenesis "Gap 1" examples compare maximal effectual buffer size against
total operand size. Reproduce the Fig. 3 benchmark table and the Fig. 11 GEMM
shape sweep with:

```bash
make gap1
```

To run the same Fig. 3 workload set on MetaX C500 hardware and collect MCPTI
DRAM/L2 transaction traffic for representative FP32 kernels:

```bash
make gap1-c500
./build/gap1_c500
```

The C500 run prints the analytical Gap 1 values plus measured DRAM bytes and
L2 transaction bytes. The Gap 1 ratio itself is a workload property, while the
traffic counters are hardware/runtime measurements for these kernels.
