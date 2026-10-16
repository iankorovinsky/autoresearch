export type Message = {
  role: "user" | "assistant";
  text: string;
  files?: string[];
};
export type Thread = {
  id: string;
  title: string;
  task: string;
  messages: Message[];
  project: string;
};
export const initialThreads: Thread[] = [
  {
    id: "matmul",
    title: "Optimize tiled matrix multiplication",
    task: "Tiled matrix multiplication",
    project: "Kernel optimization",
    messages: [
      {
        role: "user",
        text: "Optimize the matrix multiplication kernel for an NVIDIA T4. Explore different tile sizes and find the fastest configuration that passes correctness checks.",
      },
    ],
  },
  {
    id: "softmax",
    title: "Fused softmax exploration",
    task: "Fused softmax",
    project: "Kernel optimization",
    messages: [
      {
        role: "user",
        text: "Explore a fused softmax implementation. Focus on reducing memory reads and validate against the PyTorch baseline.",
      },
    ],
  },
  {
    id: "reduction",
    title: "Parallel reduction benchmarks",
    task: "Parallel reduction",
    project: "Kernel optimization",
    messages: [
      {
        role: "user",
        text: "Compare parallel reduction strategies across different input sizes on the T4.",
      },
    ],
  },
];
export const experiments = [
  {
    id: "EXP-012",
    name: "Tile 64 × 64 · shared memory",
    time: "1.24",
    speed: "2.41",
    status: "Best result",
    tile: 64,
  },
  {
    id: "EXP-011",
    name: "Tile 32 × 32 · loop unrolling",
    time: "1.48",
    speed: "2.02",
    status: "Passed",
    tile: 32,
  },
  {
    id: "EXP-010",
    name: "Tile 32 × 32 · shared memory",
    time: "1.67",
    speed: "1.79",
    status: "Passed",
    tile: 32,
  },
  {
    id: "EXP-009",
    name: "Tile 16 × 16 · shared memory",
    time: "2.12",
    speed: "1.41",
    status: "Passed",
    tile: 16,
  },
];
export const kernelCode = `import torch
from torch.utils.cpp_extension import load_inline

# EXP-012 · Best verified candidate
cuda_source = """
#define TILE_SIZE 64

__global__ void tiled_matmul(
    const float* A, const float* B,
    float* C, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float result = 0.0f;

    // Load tiles, synchronize, accumulate.
    // Full candidate available after backend connection.
}
"""

# Preview excerpt · illustrative source only
# Evaluation: 5 correctness trials / 100 timing trials`;
