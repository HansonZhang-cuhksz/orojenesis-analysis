#!/usr/bin/env python3
"""Reproduce Orojenesis Gap 1 benchmark ratios.

Gap 1 is the distance between total operand size and the maximal effectual
buffer size needed to enable full reuse.  The percentages are independent of
element width; this script prints FP32 byte counts for concreteness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ELEMENT_BYTES = 4


@dataclass(frozen=True)
class GapBenchmark:
    name: str
    total_elements: int
    effectual_elements: int
    paper_percent: float | None = None

    @property
    def total_bytes(self) -> int:
        return self.total_elements * ELEMENT_BYTES

    @property
    def effectual_bytes(self) -> int:
        return self.effectual_elements * ELEMENT_BYTES

    @property
    def percent(self) -> float:
        return 100.0 * self.effectual_elements / self.total_elements


def vadd(elements: int = 1_000_000) -> GapBenchmark:
    # Streaming vector add has no meaningful reuse; only the current element
    # triple needs to be live.
    return GapBenchmark("1m vadd", 3 * elements, 3, 0.00)


def gemv(m: int = 1000, k: int = 1000) -> GapBenchmark:
    # A is streamed, while the input and output vectors are the useful resident
    # operands for full reuse.
    return GapBenchmark("1k x 1k GEMV", m * k + k + m, k + m, 0.20)


def gemm(m: int, n: int, k: int, name: str | None = None) -> GapBenchmark:
    operands = (m * k, k * n, m * n)
    return GapBenchmark(name or f"{m}_{k}_{n} GEMM", sum(operands), min(operands))


def conv2d(
    name: str,
    p: int,
    q: int,
    c: int,
    n: int,
    r: int,
    s: int,
    input_p: int,
    input_q: int,
    paper_percent: float,
) -> GapBenchmark:
    # Tensor shapes follow the paper's compact Fig. 3 setup:
    # input[input_p, input_q, C], weights[C, N, R, S], output[P, Q, N].
    input_elements = input_p * input_q * c
    weight_elements = c * n * r * s
    output_elements = p * q * n
    operands = (input_elements, weight_elements, output_elements)
    return GapBenchmark(name, sum(operands), min(operands), paper_percent)


def fig3_benchmarks() -> list[GapBenchmark]:
    p = q = 16
    c = n = 64
    return [
        vadd(),
        gemv(),
        GapBenchmark("1k x 1k x 1k GEMM", 3 * 1000 * 1000, 1000 * 1000, 33.0),
        conv2d("1x1 conv", p, q, c, n, 1, 1, 16, 16, 11.0),
        conv2d("3x3 conv", p, q, c, n, 3, 3, 17, 17, 23.0),
        conv2d("3x3 conv stride 2", p, q, c, n, 3, 3, 32, 32, 14.0),
        conv2d("5x5 conv", p, q, c, n, 5, 5, 19, 19, 12.0),
    ]


def fig11_gemm_sweep() -> list[GapBenchmark]:
    shapes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (2048, 4096, 4096),
        (4096, 4096, 2048),
        (4096, 2048, 4096),
        (8192, 4096, 8192),
        (8192, 8192, 4096),
        (4096, 8192, 8192),
        (2048, 2048, 4096),
        (4096, 2048, 2048),
        (2048, 4096, 2048),
        (8192, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 8192, 4096),
        (2048, 8192, 8192),
        (8192, 8192, 2048),
        (8192, 2048, 8192),
        (4096, 2048, 8192),
        (4096, 8192, 2048),
        (8192, 4096, 2048),
        (8192, 2048, 4096),
        (2048, 4096, 8192),
        (2048, 8192, 4096),
        (8192, 2048, 2048),
        (2048, 2048, 8192),
        (2048, 8192, 2048),
    ]
    return [gemm(m, n, k, f"{m}_{k}_{n}") for m, n, k in shapes]


def figure_label(percent: float) -> str:
    if percent < 1.0:
        return f"{percent:.2f}%"
    return f"{round(percent):.0f}%"


def print_table(title: str, rows: Iterable[GapBenchmark], include_paper: bool) -> None:
    print(title)
    if include_paper:
        print(
            f"{'benchmark':<24} {'effectual_B':>14} {'total_B':>14} "
            f"{'repro_%':>10} {'fig_label':>10} {'paper_%':>8}"
        )
        print("-" * 84)
    else:
        print(f"{'benchmark':<24} {'effectual_B':>14} {'total_B':>14} {'repro_%':>10}")
        print("-" * 66)

    for row in rows:
        if include_paper:
            paper = "" if row.paper_percent is None else f"{row.paper_percent:>8.2f}"
            print(
                f"{row.name:<24} {row.effectual_bytes:>14} {row.total_bytes:>14} "
                f"{row.percent:>9.2f}% {figure_label(row.percent):>10} {paper}"
            )
        else:
            print(
                f"{row.name:<24} {row.effectual_bytes:>14} "
                f"{row.total_bytes:>14} {row.percent:>9.2f}%"
            )
    print()


def main() -> None:
    print_table("Fig. 3 Gap 1 reproduction", fig3_benchmarks(), include_paper=True)
    print_table("Fig. 11 GEMM Gap 1 sweep", fig11_gemm_sweep(), include_paper=False)


if __name__ == "__main__":
    main()
