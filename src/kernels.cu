#include <vector>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <math_constants.h>

#include "../tester/utils.h"

/**
 * @brief 计算两个维度的乘积，避免张量元素数量发生整数溢出
 * @param lhs 左操作数
 * @param rhs 右操作数
 * @param tensor_name 张量名称
 * @return 两个维度的乘积
 */
inline size_t checkedElementProduct(size_t lhs, size_t rhs,
                                    const char* tensor_name) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(std::string(tensor_name) +
                              " element count overflow");
  }
  return lhs * rhs;
}
/**
 * @brief 计算正数的向上取整除法
 * @param value 被除数
 * @param divisor 除数
 * @return 向上取整的除法结果
 */
__host__ __device__ __forceinline__ int positiveCeilDiv(int value,
                                                         int divisor) {
  return value / divisor + (value % divisor != 0);
}

/**
 * @brief 保存当前 GPU 的运行时硬件信息
 */
struct RuntimeDeviceInfo {
  int device_id;                 // GPU 设备编号
  int warp_size;                 // 一个 warp 包含的线程数
  int multiprocessor_count;      // GPU 多处理器数量
  int max_threads_per_block;     // 一个 block 允许的最大线程数
  size_t shared_mem_per_block;   // 每个 block 可使用的共享内存上限
};

/**
 * @brief 获取当前 GPU 的运行时信息，并按 Host 线程缓存查询结果
 * @return 当前设备对应的运行时硬件信息
 */
inline const RuntimeDeviceInfo& currentDeviceInfo() {
  int current_device = 0;
  RUNTIME_CHECK(cudaGetDevice(&current_device));

  // 每个 Host 线程独立缓存设备属性，避免每次调用算子都查询 CUDA Runtime。
  static thread_local RuntimeDeviceInfo cached{-1, 0, 0, 0, 0};
  if (cached.device_id != current_device) {
    // 当前线程切换 GPU 后重新读取硬件属性，并更新缓存中的设备编号。
    cudaDeviceProp properties{};
    RUNTIME_CHECK(cudaGetDeviceProperties(&properties, current_device));
    cached.device_id = current_device;
    cached.warp_size = properties.warpSize;
    cached.multiprocessor_count = properties.multiProcessorCount;
    cached.max_threads_per_block = properties.maxThreadsPerBlock;
    cached.shared_mem_per_block = properties.sharedMemPerBlock;
  }
  return cached;
}

/**
 * @brief 将类型 T 转换为 float，方便统一使用 FP32 计算
 * @tparam T 输入数据类型
 * @param x 待转换的数据
 * @return 转换后的 float 数据
 */
template <typename T>
__host__ __device__ __forceinline__ float toFloat(T x) {
  return static_cast<float>(x);
}

/**
 * @brief 将 half 类型转换为 float
 * @param x 待转换的 half 数据
 * @return 转换后的 float 数据
 */
template <>
__host__ __device__ __forceinline__ float toFloat<half>(half x) {
  return __half2float(x);
}

/**
 * @brief 将 FP32 计算结果转换回存储类型 T
 * @tparam T 输出数据类型
 * @param x 待转换的 float 数据
 * @return 类型为 T 的转换结果
 */
template <typename T>
__host__ __device__ __forceinline__ T fromFloat(float x) {
  return static_cast<T>(x);
}

/**
 * @brief 将 float 类型转换为 half 类型
 * @param x 待转换的 float 数据
 * @return 转换后的 half 数据
 */
template <>
__host__ __device__ __forceinline__ half fromFloat<half>(float x) {
  return __float2half(x);
}

/**
 * @brief 管理一块可复用的 GPU 显存，容量不足或设备变化时才重新分配
 * @tparam T 缓冲区中元素的数据类型
 */
template <typename T>
class ReusableDeviceBuffer {
 public:
  ReusableDeviceBuffer() = default;
  // 禁止复制
  ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
  ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;
  /**
   * @brief 对象销毁时自动释放所管理的 GPU 显存
   */
  ~ReusableDeviceBuffer() {
    if (ptr_ != nullptr) {
      int restore_device = -1;
      if (cudaGetDevice(&restore_device) == cudaSuccess) {
        // cudaFree 必须在内存所属设备上执行，释放后再恢复调用前的设备。
        if (restore_device != device_id_) {
          cudaSetDevice(device_id_);
        }
        cudaFree(ptr_);
        if (restore_device != device_id_) {
          cudaSetDevice(restore_device);
        }
      }
    }
  }

  /**
   * @brief 确保缓冲区位于指定设备并且能够容纳所需元素
   * @param count 需要容纳的元素数量
   * @param current_device 当前 GPU 设备编号
   */
  void ensure(size_t count, int current_device) {
    // 先检查 count * sizeof(T)，防止分配字节数发生无符号整数溢出。
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("device allocation size overflow");
    }

    // 设备一致且现有容量足够时直接复用，避免重复 cudaMalloc/cudaFree。
    if (device_id_ == current_device && count <= capacity_) {
      return;
    }

    // 缓冲区属于其他设备或容量不足时，先在原设备释放旧内存。
    if (ptr_ != nullptr) {
      const int restore_device = current_device;
      if (device_id_ != current_device) {
        RUNTIME_CHECK(cudaSetDevice(device_id_));
      }
      RUNTIME_CHECK(cudaFree(ptr_));
      if (device_id_ != restore_device) {
        RUNTIME_CHECK(cudaSetDevice(restore_device));
      }
      ptr_ = nullptr;
      capacity_ = 0;
      device_id_ = -1;
    }

    // 在当前设备分配新缓冲区，并记录容量及所属设备。
    RUNTIME_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    capacity_ = count;
    device_id_ = current_device;
  }
  /**
   * @brief 获取可写的 GPU 显存地址
   * @return 缓冲区首地址
   */
  T* data() { return ptr_; }

  /**
   * @brief 获取只读的 GPU 显存地址
   * @return 缓冲区首地址
   */
  const T* data() const { return ptr_; }

  /**
   * @brief 获取当前缓冲区容量
   * @return 可以容纳的 T 类型元素数量
   */
  size_t capacity() const { return capacity_; }

 private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
  int device_id_ = -1;
};

/**
 * @brief RMSNorm 计算的 Kernel
 * @tparam T 输入、权重和输出的数据类型
 * @param input GPU 输入地址
 * @param weight GPU 权重地址
 * @param output GPU 输出地址
 * @param rows 输入行数
 * @param hidden_dim 每一行的元素数量
 * @param eps 防止除零
 */
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output, size_t rows,
                              size_t hidden_dim, float eps) {
  // 一个 block 负责输入矩阵的一行，blockIdx.x 直接对应行号。
  size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  // 每个线程在共享内存中保存一个局部平方和，供后续树形归约使用。
  extern __shared__ float shm[];
  float local_sum = 0.0f;

  // block 内线程以跨步方式遍历本行，保证 hidden_dim 大于线程数时仍能覆盖全部元素。
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    float x = toFloat(input[row * hidden_dim + col]);
    local_sum += x * x;
  }

  shm[threadIdx.x] = local_sum;
  __syncthreads();

  // 树形归约：每轮将后一半线程的结果累加到前一半，最终总平方和位于 shm[0]。
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shm[threadIdx.x] += shm[threadIdx.x + stride];
    }
    __syncthreads();
  }
  // 线程 0 根据总平方和计算本行的逆 RMS，并复用 shm[0] 广播结果。
  if (threadIdx.x == 0) {
    float mean_square = shm[0] / static_cast<float>(hidden_dim);
    shm[0] = rsqrtf(mean_square + eps);
  }
  __syncthreads();

  // 同步后所有线程读取相同的 inv_rms，并行完成归一化和逐列权重缩放。
  float inv_rms = shm[0];
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    size_t idx = row * hidden_dim + col;
    float x = toFloat(input[idx]);
    float w = toFloat(weight[col]);
    output[idx] = fromFloat<T>(x * inv_rms * w);
  }
}

// -----------------------------------------------------------------------------
// FlashAttention 辅助结构与实现
// -----------------------------------------------------------------------------

/**
 * @brief 汇总所有注意力 Kernel 共用的张量形状和派生参数
 */
struct AttnParams {
  int batch_size;           // 批次大小
  int target_seq_len;       // Q 的序列长度
  int src_seq_len;          // K/V 的序列长度
  int query_heads;          // Q 的注意力头数量
  int kv_heads;             // K/V 的注意力头数量
  int head_dim;             // 每个注意力头的向量维度
  int total_queries;        // 所有 batch 中的查询总数
  int queries_per_batch;    // 每个 batch 中的查询总数
  int query_heads_per_kv;   // 每个 K/V 头对应的 Q 头数量
  bool is_causal;           // 是否使用因果掩码
};

/**
 * @brief 保存一个查询的一维索引解码结果、内存偏移和缩放信息
 */
struct QueryCoord {
  int b;                 // 当前查询所属的 batch
  int t;                 // 当前查询在目标序列中的位置
  int qh;                // 当前查询使用的 Q 头编号
  int kv_head;           // 与当前 Q 头对应的 K/V 头编号

  size_t q_base;         // 当前查询在 Q 和输出 O 中的行首偏移
  size_t kv_batch_base;  // 当前 batch 在 K/V 张量中的起始偏移

  float scale;           // 注意力分数缩放系数
  bool valid;            // 当前查询索引是否有效
};

/**
 * @brief 将一维查询索引解码为 batch、目标位置、查询头和 K/V 头
 * @param query_idx 展平后的查询索引
 * @param p 注意力形状、头数及因果掩码等参数
 * @return 当前查询对应的坐标、内存偏移和缩放信息
 */
__device__ __forceinline__ QueryCoord decodeQuery(int query_idx,
                                                   const AttnParams& p) {
  QueryCoord c;

  // Kernel 网格可能向上取整，多出的线程通过 valid 提前退出。
  c.valid = query_idx < p.total_queries;
  if (!c.valid) {
    return c;
  }

  // 将展平布局 [batch, target_position, query_head] 还原为 b、t、qh。
  c.b = query_idx / p.queries_per_batch;
  const int rem = query_idx - c.b * p.queries_per_batch;
  c.t = rem / p.query_heads;
  c.qh = rem - c.t * p.query_heads;

  // GQA 中连续的多个查询头共享同一个 K/V 头。
  c.kv_head = c.qh / p.query_heads_per_kv;

  // q_base 同时适用于布局相同的 Q 和 O；kv_batch_base 只定位到当前 batch。
  c.q_base = static_cast<size_t>(query_idx) * p.head_dim;
  c.kv_batch_base =
      static_cast<size_t>(c.b) * p.src_seq_len * p.kv_heads * p.head_dim;
  // Scaled Dot-Product Attention 使用 1/sqrt(head_dim) 缩放 Q·K。
  c.scale = rsqrtf(static_cast<float>(p.head_dim));
  return c;
}

/**
 * @brief 计算指定源位置和 K/V 头对应的行首偏移
 * @param c 当前查询的解码结果
 * @param p 注意力形状和头数参数
 * @param s K/V 在源序列中的位置
 * @return 当前 K/V 行相对于张量首地址的元素偏移
 */
__device__ __forceinline__ size_t kvRowOffset(const QueryCoord& c,
                                               const AttnParams& p, int s) {
  return c.kv_batch_base +
         (static_cast<size_t>(s) * p.kv_heads + c.kv_head) * p.head_dim;
}

/**
 * @brief 计算当前查询能够访问的源序列位置上界
 * @param c 当前查询的解码结果
 * @param p 注意力形状及因果掩码参数
 * @return 可参与注意力计算的 K/V 行数
 */
__device__ __forceinline__ int causalEnd(const QueryCoord& c,
                                          const AttnParams& p) {
  return p.is_causal ? min(c.t + 1, p.src_seq_len) : p.src_seq_len;
}

/**
 * @brief 保存 K/V 分块 Kernel 中一个查询的坐标、边界和内存偏移
 */
struct KVTiledCoord {
  int b;                    // 当前查询所属的 batch
  int qh;                   // 当前 Q 头编号
  int t0;                   // 当前查询分块的起始目标位置
  int t;                    // 当前查询在目标序列中的位置
  int kv_head;              // 当前 Q 头对应的 K/V 头编号
  int s_end;                // 当前查询可以访问的源位置上界
  int block_s_end;          // 整个查询分块需要载入的源位置上界
  size_t q_base;            // 当前查询在 Q/O 中的行首偏移
  size_t kv_batch_base;     // 当前 batch 在 K/V 中的起始偏移
  bool active;              // 当前查询是否位于有效目标序列范围内
};

/**
 * @brief 解码 K/V 分块任务并预先计算查询坐标、地址和访问边界
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam QUERIES_PER_BLOCK 每个线程块处理的查询数量
 * @param block_index 当前线程块的一维索引
 * @param query_in_block 当前查询在线程块内的编号
 * @param p 注意力形状、头数及因果掩码等参数
 * @return 当前分块查询对应的坐标和边界信息
 */
template <int HEAD_DIM, int QUERIES_PER_BLOCK>
__device__ __forceinline__ KVTiledCoord decodeKVTiledQuery(
    int block_index, int query_in_block, const AttnParams& p) {
  KVTiledCoord c;

  // 网格按 [batch, query_head, target_tile] 展平，这里先计算每层的跨度。
  const int target_tiles = positiveCeilDiv(p.target_seq_len, QUERIES_PER_BLOCK);
  const int tiles_per_batch = p.query_heads * target_tiles;
  c.b = block_index / tiles_per_batch;
  const int block_rem = block_index - c.b * tiles_per_batch;
  c.qh = block_rem / target_tiles;
  const int target_tile = block_rem - c.qh * target_tiles;
  // t0 是整个查询 tile 的起点，t 是当前线程组实际负责的查询位置。
  c.t0 = target_tile * QUERIES_PER_BLOCK;
  c.t = c.t0 + query_in_block;
  c.active = c.t < p.target_seq_len;
  // 最后一个 tile 可能不满；无效查询用 safe_t=0 构造安全地址，但不会参与计算。
  const int safe_t = c.active ? c.t : 0;
  c.kv_head = c.qh / p.query_heads_per_kv;
  c.q_base =
      (static_cast<size_t>(c.b) * p.target_seq_len * p.query_heads +
       static_cast<size_t>(safe_t) * p.query_heads + c.qh) *
      HEAD_DIM;
  c.kv_batch_base =
      static_cast<size_t>(c.b) * p.src_seq_len * p.kv_heads * HEAD_DIM;
  // s_end 是单个查询的因果边界，block_s_end 是整个 block 需要加载的最大 K/V 范围。
  c.s_end = c.active
                ? (p.is_causal ? min(c.t + 1, p.src_seq_len) : p.src_seq_len)
                : 0;
  c.block_s_end =
      p.is_causal
          ? min(min(c.t0 + QUERIES_PER_BLOCK, p.target_seq_len), p.src_seq_len)
          : p.src_seq_len;
  return c;
}

// -----------------------------------------------------------------------------
// Warp 宽度适配
// NVIDIA 使用 32 线程 warp；天数平台使用原生 64 线程 warp 和无掩码 shuffle。
// -----------------------------------------------------------------------------
#if defined(PLATFORM_ILUVATAR)
#define ATTENTION_USE_64_LANE_BACKEND 1
#endif

constexpr size_t ATTENTION_KV_TILED_MIN_QUERIES = 16384;
constexpr int ATTENTION_KV_TILED_MIN_SEQUENCE = 64;
constexpr int ATTENTION_FLOAT_KV_TILED_MIN_SEQUENCE = 128;
constexpr int ATTENTION_KV_TILE_ROWS = 64;
constexpr int ATTENTION_FLOAT_HEAD64_KV_TILE_ROWS = 32;
constexpr int ATTENTION_SCORE_TILE_ROWS = 8;

/**
 * @brief 判断当前输入规模是否适合使用 K/V 分块路径
 * @param total_queries 查询总数
 * @param target_seq_len Q 的序列长度
 * @param src_seq_len K/V 的序列长度
 * @param head_dim 每个注意力头的维度
 * @param min_sequence 启用分块路径所需的最小序列长度
 * @return 满足分块路径条件时返回 true，否则返回 false
 */
inline bool shouldUseKVTiled(size_t total_queries, int target_seq_len,
                             int src_seq_len, int head_dim,
                             int min_sequence) {
  return total_queries >= ATTENTION_KV_TILED_MIN_QUERIES &&
         target_seq_len >= min_sequence && src_seq_len >= min_sequence &&
         (head_dim == 32 || head_dim == 64);
}

/**
 * @brief 判断 half 数据是否适合使用 K/V 分块路径
 * @param total_queries 查询总数
 * @param target_seq_len Q 的序列长度
 * @param src_seq_len K/V 的序列长度
 * @param head_dim 每个注意力头的维度
 * @return 满足 half 分块路径条件时返回 true，否则返回 false
 */
inline bool shouldUseHalfKVTiled(size_t total_queries, int target_seq_len,
                                 int src_seq_len, int head_dim) {
  return shouldUseKVTiled(total_queries, target_seq_len, src_seq_len, head_dim,
                          ATTENTION_KV_TILED_MIN_SEQUENCE);
}

/**
 * @brief 判断 float 数据是否适合使用 K/V 分块路径
 * @param total_queries 查询总数
 * @param target_seq_len Q 的序列长度
 * @param src_seq_len K/V 的序列长度
 * @param head_dim 每个注意力头的维度
 * @return 满足 float 分块路径条件时返回 true，否则返回 false
 */
inline bool shouldUseFloatKVTiled(size_t total_queries, int target_seq_len,
                                  int src_seq_len, int head_dim) {
  return shouldUseKVTiled(total_queries, target_seq_len, src_seq_len, head_dim,
                          ATTENTION_FLOAT_KV_TILED_MIN_SEQUENCE);
}

/**
 * @brief 按维度顺序计算两个向量的点积，并使用 FP32 累加
 * @tparam T 输入向量的数据类型
 * @param a 第一个 GPU 向量地址
 * @param b 第二个 GPU 向量地址
 * @param n 向量元素数量
 * @return 两个向量的 FP32 点积结果
 */
template <typename T>
__device__ __forceinline__ float sequentialDotT(const T* __restrict__ a,
                                                 const T* __restrict__ b,
                                                 int n) {
  float dot = 0.0f;
  for (int d = 0; d < n; ++d) {
    dot += toFloat(a[d]) * toFloat(b[d]);
  }
  return dot;
}

/**
 * @brief 对 warp 内所有线程的局部值求和，并将结果广播给每个线程
 * @param value 当前线程的局部值
 * @return warp 内所有局部值之和
 */
__device__ __forceinline__ float warpAllReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

/**
 * @brief 对注意力线程组中的局部值求和并广播结果
 * @tparam GROUP_SIZE 注意力线程组包含的线程数量
 * @param value 当前线程的局部值
 * @return 线程组中所有局部值之和
 */
template <int GROUP_SIZE>
__device__ __forceinline__ float attentionGroupAllReduceSum(float value) {
#if defined(ATTENTION_USE_64_LANE_BACKEND)
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64,
                "Iluvatar attention groups must tile a 64-lane warp");
#pragma unroll
  for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset, GROUP_SIZE);
  }
  return value;
#else
  static_assert(GROUP_SIZE == 32, "NVIDIA attention groups are 32 lanes");
  return warpAllReduceSum(value);
#endif
}

/**
 * @brief 将指定线程的标量广播给整个注意力线程组
 * @tparam GROUP_SIZE 注意力线程组包含的线程数量
 * @param value 当前线程持有的标量
 * @param source_lane 提供广播值的线程编号
 * @return 源线程提供的标量
 */
template <int GROUP_SIZE>
__device__ __forceinline__ float attentionGroupBroadcast(float value,
                                                          int source_lane) {
#if defined(ATTENTION_USE_64_LANE_BACKEND)
  static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64,
                "Iluvatar attention groups must tile a 64-lane warp");
  return __shfl(value, source_lane, GROUP_SIZE);
#else
  static_assert(GROUP_SIZE == 32, "NVIDIA attention groups are 32 lanes");
  return __shfl_sync(0xffffffffu, value, source_lane, GROUP_SIZE);
#endif
}

// -----------------------------------------------------------------------------
// FP32 顺序累加路径：一个线程处理一个查询。
// 严格按照维度和源位置顺序累加，以匹配参考实现的数值顺序。
// 指针对齐时使用 float4 扩宽访存，但不改变元素的累加顺序。
// -----------------------------------------------------------------------------

/**
 * @brief 顺序计算两个 FP32 向量的点积，对齐时使用 float4 读取
 * @param a 第一个 GPU 向量地址
 * @param b 第二个 GPU 向量地址
 * @param n 向量元素数量
 * @return 两个向量的 FP32 点积结果
 */
__device__ __forceinline__ float sequentialDot(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                int n) {
  float dot = 0.0f;
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) &
       0xF) == 0;
  int d = 0;
  if (aligned) {
    const int n4 = n & ~3;  // 不超过 n 的最大 4 的倍数
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    for (int q = 0; q < (n4 >> 2); ++q) {
      const float4 av = a4[q];
      const float4 bv = b4[q];
      // 仍按 x、y、z、w 的顺序累加，与标量循环保持一致。
      dot += av.x * bv.x;
      dot += av.y * bv.y;
      dot += av.z * bv.z;
      dot += av.w * bv.w;
    }
    d = n4;
  }
  for (; d < n; ++d) {
    dot += a[d] * b[d];
  }
  return dot;
}

/**
 * @brief 计算编译期定长 FP32 向量的顺序点积
 * @tparam N 向量元素数量
 * @param a 第一个 GPU 向量地址
 * @param b 第二个 GPU 向量地址
 * @return 两个向量的 FP32 点积结果
 */
template <int N>
__device__ __forceinline__ float sequentialDotN(const float* __restrict__ a,
                                                 const float* __restrict__ b) {
  float dot = 0.0f;
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) &
       0xF) == 0;
  if (aligned && (N & 3) == 0) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
#pragma unroll
    for (int q = 0; q < N / 4; ++q) {
      const float4 av = a4[q];
      const float4 bv = b4[q];
      dot += av.x * bv.x;
      dot += av.y * bv.y;
      dot += av.z * bv.z;
      dot += av.w * bv.w;
    }
  } else {
#pragma unroll
    for (int d = 0; d < N; ++d) {
      dot += a[d] * b[d];
    }
  }
  return dot;
}

/**
 * @brief 使用编译期固定头维度计算 FP32 FlashAttention
 * @tparam HEAD_DIM 每个注意力头的维度
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <int HEAD_DIM>
__global__ void flashAttentionFloatKernelT(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            float* __restrict__ o,
                                            AttnParams p) {
  // 一个线程处理一个完整查询，将线性线程编号解码为 batch、位置和 Q 头。
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const QueryCoord c = decodeQuery(idx, p);
  if (!c.valid) {
    return;
  }

  // Q 与 O 使用相同的行布局；s_end 给出当前查询允许访问的 K/V 行数。
  const float scale = c.scale;
  const float* q_ptr = q + c.q_base;
  float* o_ptr = o + c.q_base;

  const int s_end = causalEnd(c, p);

  // 第一遍扫描所有可见 K，求 max(Q·K/sqrt(d))，供稳定 Softmax 减去最大值。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const float* k_ptr = k + kvRowOffset(c, p, s);
    const float dot = sequentialDotN<HEAD_DIM>(q_ptr, k_ptr);
    max_logit = fmaxf(max_logit, dot * scale);
  }

  // 没有可见 K/V 时 Softmax 无定义，按约定将整个输出向量置零。
  if (max_logit == -CUDART_INF_F) {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      o_ptr[d] = 0.0f;
    }
    return;
  }

  // 累加器大小与 HEAD_DIM 一致，常见小维度通常可以保存在寄存器中。
  __align__(16) float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  const bool vec_ok = ((reinterpret_cast<uintptr_t>(v) & 0xF) == 0);
  float4* acc4 = reinterpret_cast<float4*>(acc);

  // 第二遍重新计算分数，同时累加 Softmax 分母与未归一化的 Σ(weight·V)。
  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;

    const float dot = sequentialDotN<HEAD_DIM>(q_ptr, k_ptr);
    const float weight = expf(dot * scale - max_logit);
    denom += weight;

    if ((HEAD_DIM & 3) == 0 && vec_ok) {
      const float4* v4 = reinterpret_cast<const float4*>(v_ptr);
#pragma unroll
      for (int qi = 0; qi < HEAD_DIM / 4; ++qi) {
        const float4 vv = v4[qi];
        float4 a = acc4[qi];
        a.x += weight * vv.x;
        a.y += weight * vv.y;
        a.z += weight * vv.z;
        a.w += weight * vv.w;
        acc4[qi] = a;
      }
    } else {
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) {
        acc[d] += weight * v_ptr[d];
      }
    }
  }

  // 最后除以 Softmax 分母，得到当前查询的完整注意力输出。
  const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    o_ptr[d] = acc[d] * inv_denom;
  }
}

// -----------------------------------------------------------------------------
// 缓存 logits 的双 Kernel 路径。
// 第一个 Kernel 计算并保存 Q·K 分数和行最大值，第二个 Kernel 读取分数完成
// softmax 与 V 加权累加，从而避免重复计算点积。
// -----------------------------------------------------------------------------
/**
 * @brief 以 warp 为单位计算并缓存注意力分数及每行最大值
 * @tparam T 查询和键的数据类型
 * @tparam MAX_HEAD_DIM Kernel 支持的最大头维度
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param logits GPU 注意力分数缓存地址
 * @param row_max GPU 每个查询对应的最大分数缓存地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T, int MAX_HEAD_DIM>
__global__ void flashAttentionWarpLogitsKernel(const T* __restrict__ q,
                                                const T* __restrict__ k,
                                                float* __restrict__ logits,
                                                float* __restrict__ row_max,
                                                AttnParams p) {
  constexpr int WARP_SIZE = 32;
  constexpr int ITEMS_PER_LANE = (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

  // 一个 warp 负责一个查询；lane 表示线程在 warp 内负责的向量维度起点。
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  const int query_idx = blockIdx.x * warps_per_block + warp_in_block;
  const QueryCoord c = decodeQuery(query_idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const T* q_ptr = q + c.q_base;
  float* logit_row =logits + static_cast<size_t>(query_idx) * p.src_seq_len;

  // 将 Q 按 lane 分片缓存到寄存器，同一查询与多个 K 点积时可以重复使用。
  float q_reg[ITEMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? toFloat(q_ptr[d]) : 0.0f;
  }

  const float scale = c.scale;
  const int s_end = causalEnd(c, p);

  // 遍历可见 K：各 lane 计算部分点积，经 warp 归约得到完整注意力分数。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const T* k_ptr = k + kvRowOffset(c, p, s);

    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      const int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        dot += q_reg[i] * toFloat(k_ptr[d]);
      }
    }
    // 归约后每个 lane 都得到相同 score，仅 lane 0 写缓存，避免重复写入。
    const float score = warpAllReduceSum(dot) * scale;
    if (lane == 0) {
      logit_row[s] = score;
    }
    max_logit = fmaxf(max_logit, score);
  }
  // 为后续输出 Kernel 保存本查询所有 logits 的最大值。
  if (lane == 0) {
    row_max[query_idx] = max_logit;
  }
}

/**
 * @brief 使用单线程计算并缓存一个 FP32 查询的注意力分数及最大值
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param logits GPU 注意力分数缓存地址
 * @param row_max GPU 每个查询对应的最大分数缓存地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
__global__ void flashAttentionFloatLogitsKernel(const float* __restrict__ q,
                                                 const float* __restrict__ k,
                                                 float* __restrict__ logits,
                                                 float* __restrict__ row_max,
                                                 AttnParams p) {
  // 一个线程负责一个查询，适合 FP32 顺序点积路径。
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const QueryCoord c = decodeQuery(idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const float scale = c.scale;
  const float* q_ptr = q + c.q_base;
  float* logit_row = logits + static_cast<size_t>(idx) * p.src_seq_len;

  const int s_end = causalEnd(c, p);

  // 依次计算 Q 与每个可见 K 的缩放点积，并同步记录行最大值。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const float* k_ptr = k + kvRowOffset(c, p, s);
    const float score = sequentialDot(q_ptr, k_ptr, head_dim) * scale;
    logit_row[s] = score;
    max_logit = fmaxf(max_logit, score);
  }
  row_max[idx] = max_logit;
}

/**
 * @brief 读取缓存分数，完成 softmax 和 V 的加权累加
 * @tparam T 值张量和输出张量的数据类型
 * @tparam MAX_HEAD_DIM Kernel 支持的最大头维度
 * @param logits GPU 注意力分数缓存地址
 * @param row_max GPU 每个查询对应的最大分数缓存地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T, int MAX_HEAD_DIM>
__global__ void flashAttentionFloatOutputKernel(
    const float* __restrict__ logits, const float* __restrict__ row_max,
    const T* __restrict__ v, T* __restrict__ o, AttnParams p) {
  constexpr int WARP_SIZE = 32;
  constexpr int ITEMS_PER_LANE = (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

  // 一个 warp 消费一个查询的 logits；各 lane 分担不同的输出维度。
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  const int query_idx = blockIdx.x * warps_per_block + warp_in_block;
  const QueryCoord c = decodeQuery(query_idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const float* logit_row =
      logits + static_cast<size_t>(query_idx) * p.src_seq_len;
  const float max_logit = row_max[query_idx];
  T* o_ptr = o + c.q_base;

  const int s_end = causalEnd(c, p);

  // 每个 lane 使用寄存器保存自己负责维度上的加权 V 累加结果。
  float acc[ITEMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    acc[i] = 0.0f;
  }

  // 没有有效源位置时直接输出零。
  if (max_logit == -CUDART_INF_F) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      const int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        o_ptr[d] = fromFloat<T>(0.0f);
      }
    }
    return;
  }

  // lane 0 根据缓存分数计算 exp(score-max) 和分母，再将权重广播给整个 warp。
  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const T* v_ptr = v + kvRowOffset(c, p, s);

    // 仅线程 0 计算指数和分母，再广播给 warp 内其他线程。
    float weight = 0.0f;
    if (lane == 0) {
      weight = expf(logit_row[s] - max_logit);
      denom += weight;
    }
    weight = __shfl_sync(0xffffffffu, weight, 0);

#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      const int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        acc[i] += weight * toFloat(v_ptr[d]);
      }
    }
  }

  // 广播归一化系数，各 lane 将自己的累加结果写回对应输出维度。
  float inv_denom = 0.0f;
  if (lane == 0) {
    inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  }
  inv_denom = __shfl_sync(0xffffffffu, inv_denom, 0);
#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * WARP_SIZE;
    if (d < head_dim) {
      o_ptr[d] = fromFloat<T>(acc[i] * inv_denom);
    }
  }
}

/**
 * @brief 使用运行期头维度计算 FP32 FlashAttention
 * @tparam ACC_CAP 每个线程可使用的最大输出累加器容量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <int ACC_CAP>
__global__ void flashAttentionFloatKernel(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           float* __restrict__ o,
                                           AttnParams p) {
  // 一个线程处理一个查询；ACC_CAP 只决定寄存器数组容量，实际循环使用 head_dim。
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const QueryCoord c = decodeQuery(idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const float scale = c.scale;
  const float* q_ptr = q + c.q_base;
  float* o_ptr = o + c.q_base;

  const int s_end = causalEnd(c, p);

  // 第一遍只计算最大分数，不保存 logits，以额外点积换取更少的全局内存。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const float* k_ptr = k + kvRowOffset(c, p, s);
    const float dot = sequentialDot(q_ptr, k_ptr, head_dim);
    max_logit = fmaxf(max_logit, dot * scale);
  }

  if (max_logit == -CUDART_INF_F) {
    // 没有有效源位置时输出零。
    for (int d = 0; d < head_dim; ++d) {
      o_ptr[d] = 0.0f;
    }
    return;
  }

  // 第二遍重新计算分数，同时计算 Softmax 分母与加权 V；累加器按 16 字节对齐。
  __align__(16) float acc[ACC_CAP];
  for (int d = 0; d < head_dim; ++d) {
    acc[d] = 0.0f;
  }

  // head_dim 是 4 的倍数且 V 地址对齐时，使用 float4 向量化读取。
  const bool vec_ok = (head_dim & 3) == 0 &&
                      ((reinterpret_cast<uintptr_t>(v) & 0xF) == 0);
  const int head_dim4 = head_dim >> 2;
  float4* acc4 = reinterpret_cast<float4*>(acc);

  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;

    const float dot = sequentialDot(q_ptr, k_ptr, head_dim);
    const float weight = expf(dot * scale - max_logit);
    denom += weight;

    if (vec_ok) {
      const float4* v4 = reinterpret_cast<const float4*>(v_ptr);
      for (int q = 0; q < head_dim4; ++q) {
        const float4 vv = v4[q];
        float4 a = acc4[q];
        a.x += weight * vv.x;
        a.y += weight * vv.y;
        a.z += weight * vv.z;
        a.w += weight * vv.w;
        acc4[q] = a;
      }
    } else {
      for (int d = 0; d < head_dim; ++d) {
        acc[d] += weight * v_ptr[d];
      }
    }
  }

  // acc 当前保存未归一化的加权和，乘 1/denom 后写回 O。
  const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    o_ptr[d] = acc[d] * inv_denom;
  }
}

/**
 * @brief 使用共享内存累加器计算超大头维度的 FP32 FlashAttention
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
__global__ void flashAttentionFloatKernelSharedAcc(const float* __restrict__ q,
                                                   const float* __restrict__ k,
                                                   const float* __restrict__ v,
                                                   float* __restrict__ o,
                                                   AttnParams p) {
  // 一个线程仍负责一个查询，但把大尺寸输出累加器从线程局部数组移到共享内存。
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const QueryCoord c = decodeQuery(idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const float scale = c.scale;
  const float* q_ptr = q + c.q_base;
  float* o_ptr = o + c.q_base;

  // 每个线程占用连续 head_dim 个 float，不同线程的累加器互不重叠。
  extern __shared__ float acc_smem[];
  float* acc = acc_smem + static_cast<size_t>(threadIdx.x) * head_dim;

  const int s_end = causalEnd(c, p);

  // 第一遍扫描：计算本行最大分数。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const float* k_ptr = k + kvRowOffset(c, p, s);
    const float dot = sequentialDot(q_ptr, k_ptr, head_dim);
    max_logit = fmaxf(max_logit, dot * scale);
  }

  if (max_logit == -CUDART_INF_F) {
    for (int d = 0; d < head_dim; ++d) {
      o_ptr[d] = 0.0f;
    }
    return;
  }

  for (int d = 0; d < head_dim; ++d) {
    acc[d] = 0.0f;
  }

  // 第二遍重新计算 logits，在共享内存中累加未归一化的 weight·V。
  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;

    const float dot = sequentialDot(q_ptr, k_ptr, head_dim);
    const float weight = expf(dot * scale - max_logit);
    denom += weight;

    for (int d = 0; d < head_dim; ++d) {
      acc[d] += weight * v_ptr[d];
    }
  }

  // 将共享内存中的累加结果归一化并写回当前查询对应的 O 行。
  const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    o_ptr[d] = acc[d] * inv_denom;
  }
}
// warp 协作在线 softmax 路径：每个线程负责若干 head_dim 分量。
// MAX_HEAD_DIM 是编译期容量档位，用来限制每个线程持有的寄存器数组大小。
// -----------------------------------------------------------------------------
/**
 * @brief 使用 warp 协作和寄存器累加器计算在线 Softmax FlashAttention
 * @tparam T 输入和输出张量的数据类型
 * @tparam MAX_HEAD_DIM Kernel 支持的最大头维度
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T, int MAX_HEAD_DIM>
__global__ void flashAttentionWarpKernelRegister(const T* __restrict__ q,
                                                 const T* __restrict__ k,
                                                 const T* __restrict__ v,
                                                 T* __restrict__ o,
                                                 AttnParams p) {
  constexpr int WARP_SIZE = 32;
  constexpr int ITEMS_PER_LANE = (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  // 同一 warp 的线程得到相同 query_idx，因此这里会整 warp 一起退出。
  const int query_idx = blockIdx.x * warps_per_block + warp_in_block;
  const QueryCoord c = decodeQuery(query_idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const T* q_ptr = q + c.q_base;
  T* o_ptr = o + c.q_base;

  // Q 和输出累加器都按 lane 分片保存在寄存器中，避免共享内存访问。
  float q_reg[ITEMS_PER_LANE];
  float acc[ITEMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? toFloat(q_ptr[d]) : 0.0f;
    acc[i] = 0.0f;
  }

  const float scale = c.scale;
  const int s_end = causalEnd(c, p);

  // 单遍在线 softmax：维护动态最大值和分母，最大值变化时同步缩放已有累加结果。
  float m = -CUDART_INF_F;  // 当前最大分数
  float l = 0.0f;           // 当前 softmax 分母
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const T* k_ptr = k + kv_base;
    const T* v_ptr = v + kv_base;

    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      const int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        dot += q_reg[i] * toFloat(k_ptr[d]);
      }
    }
    const float score = warpAllReduceSum(dot) * scale;

    // 在线 Softmax 更新：若最大值增大，用 alpha 重缩放旧分母和旧输出。
    const float m_new = fmaxf(m, score);
    float alpha = 0.0f;
    float beta = 0.0f;
    if (lane == 0) {
      alpha = (m == -CUDART_INF_F) ? 0.0f : expf(m - m_new);
      beta = expf(score - m_new);
      l = l * alpha + beta;
    }
    m = m_new;
    alpha = __shfl_sync(0xffffffffu, alpha, 0);
    beta = __shfl_sync(0xffffffffu, beta, 0);

#pragma unroll
    // beta 是当前 K/V 的未归一化权重，各 lane 并行更新自己负责的 V 维度。
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      const int d = lane + i * WARP_SIZE;
      if (d < head_dim) {
        acc[i] = acc[i] * alpha + beta * toFloat(v_ptr[d]);
      }
    }
  }

  // 循环结束后只需除以在线维护的分母 l，无需再次遍历 K/V。
  float inv_denom = 0.0f;
  if (lane == 0) {
    inv_denom = (l > 0.0f) ? (1.0f / l) : 0.0f;
  }
  inv_denom = __shfl_sync(0xffffffffu, inv_denom, 0);
#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * WARP_SIZE;
    if (d < head_dim) {
      o_ptr[d] = fromFloat<T>(acc[i] * inv_denom);
    }
  }
}

/**
 * @brief 对 K/V 分块并在同一线程块内复用，以计算 FlashAttention
 * @tparam T 输入和输出张量的数据类型
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam GROUP_SIZE 每个查询使用的线程数
 * @tparam GROUPS_PER_BLOCK 每个线程块处理的查询组数
 * @tparam KV_TILE 每次载入共享内存的 K/V 行数
 * @tparam SCORE_TILE 每组缓存的注意力分数数量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T, int HEAD_DIM, int GROUP_SIZE, int GROUPS_PER_BLOCK,
          int KV_TILE, int SCORE_TILE>
__global__ void flashAttentionKVTiledKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ o, AttnParams p) {
  static_assert(HEAD_DIM % GROUP_SIZE == 0 || HEAD_DIM < GROUP_SIZE,
                "tiled head dimension must fit the group layout");
  constexpr int ITEMS_PER_LANE =
      (HEAD_DIM + GROUP_SIZE - 1) / GROUP_SIZE;

  // block 被划分为多个线程组：每组处理一个查询，组内 lane 分担 head_dim 维度。
  const int lane = threadIdx.x % GROUP_SIZE;
  const int group_in_block = threadIdx.x / GROUP_SIZE;
  const KVTiledCoord c =
      decodeKVTiledQuery<HEAD_DIM, GROUPS_PER_BLOCK>(blockIdx.x,
                                                     group_in_block, p);

  // 每个 lane 读取步长为 GROUP_SIZE 的 Q 分量，并为相同维度准备输出累加器。
  float q_values[ITEMS_PER_LANE];
  float acc[ITEMS_PER_LANE];
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const int d = lane + item * GROUP_SIZE;
    q_values[item] =
        (c.active && d < HEAD_DIM) ? toFloat(q[c.q_base + d]) : 0.0f;
    acc[item] = 0.0f;
  }
  // m、l 分别保存跨 K/V tile 的在线 Softmax 最大值和分母。
  float m = -CUDART_INF_F;
  float l = 0.0f;
  const float scale = rsqrtf(static_cast<float>(p.head_dim));

  // block 中所有查询组共享同一份 K/V tile，减少对全局内存的重复读取。
  __shared__ T k_tile[KV_TILE * HEAD_DIM];
  __shared__ T v_tile[KV_TILE * HEAD_DIM];

  // FP32 按参考顺序分三步处理每个 tile：最大值、分母、归一化输出。
  if constexpr (std::is_same<T, float>::value) {
    // FP32 路径将每小块分数写入共享内存，由 lane 0 按固定顺序累加，保持数值顺序稳定。
    __shared__ float ordered_scores[GROUPS_PER_BLOCK][SCORE_TILE];
    __shared__ float ordered_q[GROUPS_PER_BLOCK][HEAD_DIM];

    // 先把各 lane 的 Q 分片合并成连续行，便于后续顺序点积。
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
      const int d = lane + item * GROUP_SIZE;
      if (d < HEAD_DIM) {
        ordered_q[group_in_block][d] = q_values[item];
      }
    }
    __syncthreads();

    // 沿源序列按 KV_TILE 行推进；每轮只在共享内存中保留当前 K/V 分块。
    for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
      const int rows = min(KV_TILE, c.block_s_end - s0);
      // 协作加载当前 K/V tile 到共享内存。
      for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
        const int row = idx / HEAD_DIM;
        const int d = idx - row * HEAD_DIM;
        const size_t source =
            c.kv_batch_base +
            (static_cast<size_t>(s0 + row) * p.kv_heads + c.kv_head) *
                HEAD_DIM +
            d;
        k_tile[idx] = k[source];
        v_tile[idx] = v[source];
      }
      __syncthreads();

      // 因果模式下，同一 block 内不同查询的有效 K/V 行数可能不同。
      const int query_rows =
          c.active ? max(0, min(rows, c.s_end - s0)) : 0;
      // 第一遍计算当前 tile 的最大分数。
      float row_max = -CUDART_INF_F;
      for (int row0 = 0; row0 < rows; row0 += SCORE_TILE) {
        const int score_rows = min(SCORE_TILE, rows - row0);
        if (lane < score_rows) {
          float dot = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            dot += ordered_q[group_in_block][d] *
                   k_tile[(row0 + lane) * HEAD_DIM + d];
          }
          ordered_scores[group_in_block][lane] = dot * scale;
        }
        __syncthreads();

        const int valid_rows = max(0, min(score_rows, query_rows - row0));
        if (lane == 0) {
          for (int row = 0; row < valid_rows; ++row) {
            row_max = fmaxf(row_max, ordered_scores[group_in_block][row]);
          }
        }
        __syncthreads();
      }
      row_max = attentionGroupBroadcast<GROUP_SIZE>(row_max, 0);

      if (row_max != -CUDART_INF_F) {
        // 第二遍计算当前 tile 的 softmax 分母。
        float tile_sum = 0.0f;
        for (int row0 = 0; row0 < rows; row0 += SCORE_TILE) {
          const int score_rows = min(SCORE_TILE, rows - row0);
          if (lane < score_rows) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
              dot += ordered_q[group_in_block][d] *
                     k_tile[(row0 + lane) * HEAD_DIM + d];
            }
            ordered_scores[group_in_block][lane] = dot * scale;
          }
          __syncthreads();

          const int valid_rows = max(0, min(score_rows, query_rows - row0));
          if (lane == 0) {
            for (int row = 0; row < valid_rows; ++row) {
              tile_sum +=
                  expf(ordered_scores[group_in_block][row] - row_max);
            }
          }
          __syncthreads();
        }
        tile_sum = attentionGroupBroadcast<GROUP_SIZE>(tile_sum, 0);

        // 将当前 tile 的 (row_max, tile_sum) 与之前 tile 的 (m, l) 合并。
        const float new_max = fmaxf(m, row_max);
        const float old_factor = expf(m - new_max);
        const float tile_factor = expf(row_max - new_max);
        const float new_sum = l * old_factor + tile_sum * tile_factor;
        // 最大值或分母发生变化后，旧的归一化输出需要按新基准重新缩放。
        const float old_output_scale = (l / new_sum) * old_factor;
#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
          acc[item] *= old_output_scale;
        }

        // 第三遍计算归一化概率并累加 V。
        for (int row0 = 0; row0 < rows; row0 += SCORE_TILE) {
          const int score_rows = min(SCORE_TILE, rows - row0);
          if (lane < score_rows) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
              dot += ordered_q[group_in_block][d] *
                     k_tile[(row0 + lane) * HEAD_DIM + d];
            }
            ordered_scores[group_in_block][lane] = dot * scale;
          }
          __syncthreads();

          const int valid_rows = max(0, min(score_rows, query_rows - row0));
          for (int row = 0; row < valid_rows; ++row) {
            float probability = 0.0f;
            if (lane == 0) {
              probability =
                  expf(ordered_scores[group_in_block][row] - new_max) /
                  new_sum;
            }
            probability =
                attentionGroupBroadcast<GROUP_SIZE>(probability, 0);
#pragma unroll
            for (int item = 0; item < ITEMS_PER_LANE; ++item) {
              const int d = lane + item * GROUP_SIZE;
              if (d < HEAD_DIM) {
                acc[item] +=
                    probability * v_tile[(row0 + row) * HEAD_DIM + d];
              }
            }
          }
          __syncthreads();
        }

        m = new_max;
        l = new_sum;
      }
      __syncthreads();
    }
  } else {
    // half 路径使用单遍在线 softmax，减少 Q·K 的重复计算。
    for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
    // 所有线程协作把当前源序列分块的 K/V 搬入共享内存。
    const int rows = min(KV_TILE, c.block_s_end - s0);
    for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
      const int row = idx / HEAD_DIM;
      const int d = idx - row * HEAD_DIM;
      const size_t source =
          c.kv_batch_base +
          (static_cast<size_t>(s0 + row) * p.kv_heads + c.kv_head) * HEAD_DIM +
          d;
      k_tile[idx] = k[source];
      v_tile[idx] = v[source];
    }
    __syncthreads();

    const int query_rows =
        c.active ? max(0, min(rows, c.s_end - s0)) : 0;
    // 再将 K/V tile 切成 SCORE_TILE 行的小块，限制每个线程的临时分数数组大小。
    for (int row0 = 0; row0 < query_rows; row0 += SCORE_TILE) {
      const int score_rows = min(SCORE_TILE, query_rows - row0);
      float scores[SCORE_TILE];
#pragma unroll
      for (int row = 0; row < SCORE_TILE; ++row) {
        scores[row] = -CUDART_INF_F;
      }
      // 每个 lane 计算 Q·K 的部分和，线程组归约得到每一行的完整分数。
      for (int row = 0; row < score_rows; ++row) {
        float dot = 0.0f;
#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
          const int d = lane + item * GROUP_SIZE;
          if (d < HEAD_DIM) {
            dot += q_values[item] *
                   toFloat(k_tile[(row0 + row) * HEAD_DIM + d]);
          }
        }
        scores[row] = attentionGroupAllReduceSum<GROUP_SIZE>(dot) * scale;
      }
      {
        const int valid_rows = max(0, min(score_rows, query_rows - row0));
        float tile_max = -CUDART_INF_F;
        for (int row = 0; row < valid_rows; ++row) {
          tile_max = fmaxf(tile_max, scores[row]);
        }
        // 先用新最大值缩放历史状态，再逐行加入当前小块的 exp(score-new_max)。
        const float new_max = fmaxf(m, tile_max);
        float alpha = 0.0f;
        if (lane == 0) {
          alpha = m == -CUDART_INF_F ? 0.0f : expf(m - new_max);
          l *= alpha;
        }
        m = new_max;
        alpha = attentionGroupBroadcast<GROUP_SIZE>(alpha, 0);
#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
          acc[item] *= alpha;
        }

        // 每个分数对应同一行 V；beta 广播后各 lane 更新自己负责的输出维度。
        for (int row = 0; row < valid_rows; ++row) {
          float beta = 0.0f;
          if (lane == 0) {
            beta = expf(scores[row] - new_max);
            l += beta;
          }
          beta = attentionGroupBroadcast<GROUP_SIZE>(beta, 0);
#pragma unroll
          for (int item = 0; item < ITEMS_PER_LANE; ++item) {
            const int d = lane + item * GROUP_SIZE;
            if (d < HEAD_DIM) {
              acc[item] +=
                  beta * toFloat(v_tile[(row0 + row) * HEAD_DIM + d]);
            }
          }
        }
      }
    }
    __syncthreads();
  }
  }

  // FP32 路径在 tile 合并时已经保持归一化；half 路径最后统一除以分母 l。
  // 仅有效查询写回，避免最后一个未填满的查询 tile 越界。
  if (c.active) {
    float output_scale = 1.0f;
    if constexpr (!std::is_same<T, float>::value) {
      if (lane == 0) {
        output_scale = (l > 0.0f) ? (1.0f / l) : 0.0f;
      }
      output_scale =
          attentionGroupBroadcast<GROUP_SIZE>(output_scale, 0);
    }
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
      const int d = lane + item * GROUP_SIZE;
      if (d < HEAD_DIM) {
        o[c.q_base + d] = fromFloat<T>(acc[item] * output_scale);
      }
    }
  }
}

/**
 * @brief 对查询和 K/V 同时分块并复用共享内存，以计算 FP32 FlashAttention
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam QUERY_TILE 每个线程块处理的查询数量
 * @tparam KV_TILE 每次载入共享内存的 K/V 行数
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <int HEAD_DIM, int QUERY_TILE, int KV_TILE>
__global__ void flashAttentionFloatQueryTiledKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, AttnParams p) {
  static_assert(HEAD_DIM == 32,
                "the reference-compatible query tile is specialized for D32");
  // 动态共享内存前半部分存 K tile，后半部分存相同行范围的 V tile。
  extern __shared__ float tile[];
  float* k_tile = tile;
  float* v_tile = tile + KV_TILE * HEAD_DIM;

  // blockIdx.x 解码为 batch、Q 头和查询 tile；block 内每个线程负责一个查询位置。
  const int target_tiles =
      positiveCeilDiv(p.target_seq_len, QUERY_TILE);
  const int tiles_per_batch = p.query_heads * target_tiles;
  const int b = blockIdx.x / tiles_per_batch;
  const int block_rem = blockIdx.x - b * tiles_per_batch;
  const int qh = block_rem / target_tiles;
  const int target_tile = block_rem - qh * target_tiles;
  const int t = target_tile * QUERY_TILE + threadIdx.x;
  const int tile_query_end = static_cast<int>(min(
      static_cast<long long>(p.target_seq_len),
      (static_cast<long long>(target_tile) + 1) * QUERY_TILE));
  // 因果模式只需加载当前查询 tile 末尾之前的 K/V，避免访问未来位置。
  const int block_source_end =
      p.is_causal ? min(tile_query_end, p.src_seq_len) : p.src_seq_len;
  const bool active = t < p.target_seq_len;
  const int kv_head = qh / p.query_heads_per_kv;
  const size_t q_base =
      (static_cast<size_t>(b) * p.target_seq_len * p.query_heads +
       static_cast<size_t>(active ? t : 0) * p.query_heads + qh) *
      HEAD_DIM;

  // 每个线程把自己的完整 Q 保存在寄存器，并维护一个完整输出向量。
  float q_values[HEAD_DIM];
  float output[HEAD_DIM];
  if (active) {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      q_values[d] = q[q_base + d];
    }
  }
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    output[d] = 0.0f;
  }

  // running_max/running_sum 保存已处理 K/V tile 的在线 Softmax 状态。
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
  const size_t kv_batch_base =
      static_cast<size_t>(b) * p.src_seq_len * p.kv_heads * HEAD_DIM;

  for (int s0 = 0; s0 < block_source_end; s0 += KV_TILE) {
    const int rows = min(KV_TILE, block_source_end - s0);
    // block 内所有查询协作加载并复用 K/V tile。
    for (int idx = threadIdx.x; idx < rows * HEAD_DIM;
         idx += blockDim.x) {
      const int row = idx / HEAD_DIM;
      const int d = idx - row * HEAD_DIM;
      const size_t source =
          kv_batch_base +
          (static_cast<size_t>(s0 + row) * p.kv_heads + kv_head) *
              HEAD_DIM +
          d;
      k_tile[row * HEAD_DIM + d] = k[source];
      v_tile[row * HEAD_DIM + d] = v[source];
    }
    __syncthreads();

    if (active) {
      // 计算 tile 最大值，并用在线公式合并到已有 softmax 状态。
      float tile_max = -CUDART_INF_F;
      for (int row = 0; row < rows; ++row) {
        const int s = s0 + row;
        if (p.is_causal && s > t) {
          continue;
        }
        float score = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          score += q_values[d] * k_tile[row * HEAD_DIM + d];
        }
        score *= scale;
        if (score > tile_max) {
          tile_max = score;
        }
      }

      if (tile_max != -CUDART_INF_F) {
        // 第二遍计算当前 tile 相对 tile_max 的指数和，避免指数上溢。
        float tile_sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
          const int s = s0 + row;
          if (p.is_causal && s > t) {
            continue;
          }
          float score = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            score += q_values[d] * k_tile[row * HEAD_DIM + d];
          }
          score *= scale;
          tile_sum += expf(score - tile_max);
        }

        // 把当前 tile 的 Softmax 状态合并到之前的状态，并重缩放已有输出。
        const float new_max = fmaxf(running_max, tile_max);
        const float old_factor = expf(running_max - new_max);
        const float new_sum =
            running_sum * old_factor +
            tile_sum * expf(tile_max - new_max);
        const float old_output_scale =
            (running_sum / new_sum) * old_factor;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          output[d] *= old_output_scale;
        }

        // 第三遍得到相对于合并后分母的概率，并累加 probability·V。
        for (int row = 0; row < rows; ++row) {
          const int s = s0 + row;
          if (p.is_causal && s > t) {
            continue;
          }
          float score = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            score += q_values[d] * k_tile[row * HEAD_DIM + d];
          }
          score *= scale;
          const float probability = expf(score - new_max) / new_sum;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            output[d] += probability * v_tile[row * HEAD_DIM + d];
          }
        }

        running_max = new_max;
        running_sum = new_sum;
      }
    }
    __syncthreads();
  }

  // 最后一个查询 tile 可能不满，仅有效线程写回自己的输出行。
  if (active) {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      o[q_base + d] = output[d];
    }
  }
}

/**
 * @brief 启动 head_dim=32 的 FP32 查询分块 FlashAttention Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状、头数及因果掩码等参数
 */
inline void launchFloatD32QueryTiledAttention(
    const float* d_q, const float* d_k, const float* d_v, float* d_o,
    const AttnParams& params) {
  constexpr int HEAD_DIM = 32;
  constexpr int QUERY_TILE = 128;
  constexpr int KV_TILE = 64;
  const int target_tiles =
      positiveCeilDiv(params.target_seq_len, QUERY_TILE);
  const int blocks = params.batch_size * params.query_heads * target_tiles;
  const size_t shared_mem =
      2ull * KV_TILE * HEAD_DIM * sizeof(float);
  flashAttentionFloatQueryTiledKernel<HEAD_DIM, QUERY_TILE, KV_TILE>
      <<<blocks, QUERY_TILE, shared_mem>>>(d_q, d_k, d_v, d_o, params);
}

/**
 * @brief 逐元素计算 FlashAttention 输出的通用回退 Kernel
 * @tparam T 输入和输出张量的数据类型
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param output_elements 输出张量的元素总数
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T>
__global__ void flashAttentionElementwiseFallbackKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ o, size_t output_elements,
    AttnParams p) {
  // 网格步进循环让有限数量的线程覆盖整个输出张量，避免网格维度过大。
  const size_t thread_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t thread_stride =
      static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t output_index = thread_index; output_index < output_elements;
       output_index += thread_stride) {
    // 将输出元素拆成“所属查询”和“头内维度”，每次只计算一个标量结果。
    const int query_index =
        static_cast<int>(output_index / static_cast<size_t>(p.head_dim));
    const int output_dim =
        static_cast<int>(output_index % static_cast<size_t>(p.head_dim));
    const QueryCoord c = decodeQuery(query_index, p);
    const T* q_ptr = q + c.q_base;
    const int s_end = causalEnd(c, p);

    // 第一遍扫描全部可见 K，求稳定 Softmax 所需的最大分数。
    float maximum = -CUDART_INF_F;
    for (int s = 0; s < s_end; ++s) {
      const T* k_ptr = k + kvRowOffset(c, p, s);
      const float score = sequentialDotT<T>(q_ptr, k_ptr, p.head_dim) * c.scale;
      maximum = fmaxf(maximum, score);
    }

    if (maximum == -CUDART_INF_F) {
      o[output_index] = fromFloat<T>(0.0f);
      continue;
    }

    // 第二遍只累加当前 output_dim 对应的 V 分量，因此不需要 head_dim 大小的累加器。
    float denominator = 0.0f;
    float result = 0.0f;
    for (int s = 0; s < s_end; ++s) {
      const size_t kv_offset = kvRowOffset(c, p, s);
      const T* k_ptr = k + kv_offset;
      const T* v_ptr = v + kv_offset;
      const float score = sequentialDotT<T>(q_ptr, k_ptr, p.head_dim) * c.scale;
      const float weight = expf(score - maximum);
      denominator += weight;
      result += weight * toFloat(v_ptr[output_dim]);
    }
    o[output_index] = fromFloat<T>(denominator > 0.0f
                                       ? (result / denominator)
                                       : 0.0f);
  }
}

/**
 * @brief 根据输出规模和 SM 数量启动逐元素 FlashAttention 回退 Kernel
 * @tparam T 输入和输出张量的数据类型
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param output_elements 输出张量的元素总数
 * @param params 注意力形状、头数及因果掩码等参数
 * @param device 当前 GPU 的运行时硬件信息
 */
template <typename T>
inline void launchElementwiseAttentionFallback(
    const T* d_q, const T* d_k, const T* d_v, T* d_o,
    size_t output_elements, const AttnParams& params,
    const RuntimeDeviceInfo& device) {
  // 线程数不超过设备上限；block 数同时受输出规模和经验性占用率上限约束。
  const int threads = std::max(1, std::min(256, device.max_threads_per_block));
  const size_t required_blocks =
      (output_elements + static_cast<size_t>(threads) - 1) / threads;
  const size_t occupancy_blocks = static_cast<size_t>(
      std::max(1, device.multiprocessor_count * 8));
  const int blocks = static_cast<int>(std::min(required_blocks,
                                               occupancy_blocks));
  flashAttentionElementwiseFallbackKernel<T><<<blocks, threads>>>(
      d_q, d_k, d_v, d_o, output_elements, params);
}

/**
 * @brief 使用共享内存保存查询和累加器的 warp 级 FlashAttention 回退 Kernel
 * @tparam T 输入和输出张量的数据类型
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T>
__global__ void flashAttentionWarpKernelShared(const T* __restrict__ q,
                                               const T* __restrict__ k,
                                               const T* __restrict__ v,
                                               T* __restrict__ o,
                                               AttnParams p) {
  constexpr int WARP_SIZE = 32;

  // 一个 warp 处理一个查询，lane 以步长 32 遍历该查询的头内维度。
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  const int query_idx = blockIdx.x * warps_per_block + warp_in_block;
  const QueryCoord c = decodeQuery(query_idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;

  // 共享内存布局：先存每个 warp 的 Q，随后存对应的输出累加器。
  extern __shared__ float smem[];
  const size_t per_kind = static_cast<size_t>(warps_per_block) * head_dim;
  float* q_cache = smem + static_cast<size_t>(warp_in_block) * head_dim;
  float* acc = smem + per_kind + static_cast<size_t>(warp_in_block) * head_dim;

  const T* q_ptr = q + c.q_base;
  T* o_ptr = o + c.q_base;

  // 各 lane 协作缓存完整 Q，并将同样布局的输出累加器清零。
  for (int d = lane; d < head_dim; d += WARP_SIZE) {
    q_cache[d] = toFloat(q_ptr[d]);
    acc[d] = 0.0f;
  }
  __syncwarp();

  const float scale = c.scale;
  const int s_end = causalEnd(c, p);

  // 单遍在线 softmax：线程 0 更新分母和缩放系数，再广播给整个 warp。
  float m = -CUDART_INF_F;
  float l = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const T* k_ptr = k + kv_base;
    const T* v_ptr = v + kv_base;

    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
      dot += q_cache[d] * toFloat(k_ptr[d]);
    }
    const float score = warpAllReduceSum(dot) * scale;

    // 在线 Softmax 同时更新最大值、分母和 V 累加器，不保存中间 logits。
    const float m_new = fmaxf(m, score);
    float alpha = 0.0f;
    float beta = 0.0f;
    if (lane == 0) {
      alpha = (m == -CUDART_INF_F) ? 0.0f : expf(m - m_new);
      beta = expf(score - m_new);
      l = l * alpha + beta;
    }
    m = m_new;
    alpha = __shfl_sync(0xffffffffu, alpha, 0);
    beta = __shfl_sync(0xffffffffu, beta, 0);

    for (int d = lane; d < head_dim; d += WARP_SIZE) {
      acc[d] = acc[d] * alpha + beta * toFloat(v_ptr[d]);
    }
  }

  // lane 0 计算归一化系数，广播后各 lane 写回自己负责的输出维度。
  float inv_denom = 0.0f;
  if (lane == 0) {
    inv_denom = (l > 0.0f) ? (1.0f / l) : 0.0f;
  }
  inv_denom = __shfl_sync(0xffffffffu, inv_denom, 0);
  for (int d = lane; d < head_dim; d += WARP_SIZE) {
    o_ptr[d] = fromFloat<T>(acc[d] * inv_denom);
  }
}

// -----------------------------------------------------------------------------
// 可移植路径：一个线程处理一个查询，不依赖 warp 原语，适配非 32 线程 warp。
// 每个线程的 FP32 累加器位于共享内存中，使用两遍稳定 softmax。
// -----------------------------------------------------------------------------
/**
 * @brief 使用单线程处理每个查询的可移植 FlashAttention Kernel
 * @tparam T 输入和输出张量的数据类型
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T>
__global__ void flashAttentionThreadPerQueryKernel(const T* __restrict__ q,
                                                    const T* __restrict__ k,
                                                    const T* __restrict__ v,
                                                    T* __restrict__ o,
                                                    AttnParams p) {
  // 一个线程独立处理一个查询，不使用任何 warp 宽度相关原语。
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const QueryCoord c = decodeQuery(idx, p);
  if (!c.valid) {
    return;
  }

  const int head_dim = p.head_dim;
  const float scale = c.scale;
  const T* q_ptr = q + c.q_base;
  T* o_ptr = o + c.q_base;

  // 每个线程在共享内存中获得 head_dim 个 FP32 累加元素，适配运行期头维度。
  extern __shared__ float tpq_smem[];
  float* acc = tpq_smem + static_cast<size_t>(threadIdx.x) * head_dim;

  const int s_end = causalEnd(c, p);

  // 第一遍求最大分数，第二遍求 Softmax 分母和 V 加权和，保证数值稳定。
  float max_logit = -CUDART_INF_F;
  for (int s = 0; s < s_end; ++s) {
    const T* k_ptr = k + kvRowOffset(c, p, s);
    const float dot = sequentialDotT<T>(q_ptr, k_ptr, head_dim);
    max_logit = fmaxf(max_logit, dot * scale);
  }

  if (max_logit == -CUDART_INF_F) {
    for (int d = 0; d < head_dim; ++d) {
      o_ptr[d] = fromFloat<T>(0.0f);
    }
    return;
  }

  for (int d = 0; d < head_dim; ++d) {
    acc[d] = 0.0f;
  }

  // 第二遍扫描：计算分母和加权 V。
  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base = kvRowOffset(c, p, s);
    const T* k_ptr = k + kv_base;
    const T* v_ptr = v + kv_base;

    const float dot = sequentialDotT<T>(q_ptr, k_ptr, head_dim);
    const float weight = expf(dot * scale - max_logit);
    denom += weight;

    for (int d = 0; d < head_dim; ++d) {
      acc[d] += weight * toFloat(v_ptr[d]);
    }
  }

  // 归一化共享内存中的累加结果，并转换回输出数据类型。
  const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    o_ptr[d] = fromFloat<T>(acc[d] * inv_denom);
  }
}

/**
 * @brief 使用可配置线程组和共享内存归约计算 FlashAttention
 * @tparam T 输入和输出张量的数据类型
 * @tparam GROUP_SIZE 每个查询使用的线程数
 * @tparam GROUPS_PER_BLOCK 每个线程块处理的查询组数
 * @tparam MAX_HEAD_DIM Kernel 支持的最大头维度
 * @tparam SCORE_TILE 每组缓存的注意力分数数量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状、头数及因果掩码等参数
 */
template <typename T, int GROUP_SIZE, int GROUPS_PER_BLOCK, int MAX_HEAD_DIM,
          int SCORE_TILE>
__global__ void flashAttentionPortableGroupKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ o, AttnParams p) {
  static_assert(GROUP_SIZE >= SCORE_TILE,
                "score tile needs at least one lane per row");
  constexpr int ITEMS_PER_LANE =
      (MAX_HEAD_DIM + GROUP_SIZE - 1) / GROUP_SIZE;

  // partial 保存各 lane 的点积部分和；scores 保存归约后的完整缩放分数。
  // 最后一维多分配一个元素，用于降低共享内存 bank 冲突风险。
  __shared__ float partial[GROUPS_PER_BLOCK][SCORE_TILE][GROUP_SIZE + 1];
  __shared__ float scores[GROUPS_PER_BLOCK][SCORE_TILE];

  // 三维网格直接映射 batch、目标位置和 Q 头组，组内 lane 分担 head_dim。
  const int lane = threadIdx.x % GROUP_SIZE;
  const int group = threadIdx.x / GROUP_SIZE;
  const int b = blockIdx.z;
  const int t = blockIdx.y;
  const int qh = blockIdx.x * GROUPS_PER_BLOCK + group;
  const bool active = qh < p.query_heads;
  const int safe_qh = active ? qh : 0;
  const int kv_head = safe_qh / p.query_heads_per_kv;

  const size_t q_base =
      (static_cast<size_t>(b) * p.target_seq_len * p.query_heads +
       static_cast<size_t>(t) * p.query_heads + safe_qh) *
      p.head_dim;
  const size_t kv_batch_base =
      static_cast<size_t>(b) * p.src_seq_len * p.kv_heads * p.head_dim;
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * p.head_dim;
  const T* q_ptr = q + q_base;
  T* o_ptr = o + q_base;

  // 每个 lane 将自己的 Q 分片和输出分片保存在寄存器中。
  float q_values[ITEMS_PER_LANE];
  float acc[ITEMS_PER_LANE];
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const int d = lane + item * GROUP_SIZE;
    q_values[item] = d < p.head_dim ? toFloat(q_ptr[d]) : 0.0f;
    acc[item] = 0.0f;
  }

  const float scale = rsqrtf(static_cast<float>(p.head_dim));
  const int s_end =
      p.is_causal ? min(t + 1, p.src_seq_len) : p.src_seq_len;
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  // 源序列按 SCORE_TILE 行处理，避免一次为全部 logits 分配缓存。
  for (int s0 = 0; s0 < s_end; s0 += SCORE_TILE) {
    const int rows = min(SCORE_TILE, s_end - s0);
    for (int row = 0; row < rows; ++row) {
      const T* k_ptr =
          k + kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
          static_cast<size_t>(kv_head) * p.head_dim;
      float dot = 0.0f;
#pragma unroll
      for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int d = lane + item * GROUP_SIZE;
        if (d < p.head_dim) {
          dot += q_values[item] * toFloat(k_ptr[d]);
        }
      }
      partial[group][row][lane] = dot;
    }
    // 等待所有 lane 写完部分点积后，由前 rows 个 lane 分别归约一行分数。
    __syncthreads();

    if (lane < rows) {
      float dot = 0.0f;
#pragma unroll
      for (int source_lane = 0; source_lane < GROUP_SIZE; ++source_lane) {
        dot += partial[group][lane][source_lane];
      }
      scores[group][lane] = dot * scale;
    }
    __syncthreads();

    // 将当前分数块合并进在线 Softmax 状态，并同步缩放历史输出。
    float tile_max = -CUDART_INF_F;
    // 逐行计算新基准下的权重 beta，并累加对应 V 分片。
    for (int row = 0; row < rows; ++row) {
      tile_max = fmaxf(tile_max, scores[group][row]);
    }
    const float new_max = fmaxf(running_max, tile_max);
    const float alpha =
        running_max == -CUDART_INF_F ? 0.0f : expf(running_max - new_max);
    running_max = new_max;
    running_sum *= alpha;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
      acc[item] *= alpha;
    }

    for (int row = 0; row < rows; ++row) {
      const float beta = expf(scores[group][row] - new_max);
      running_sum += beta;
      const T* v_ptr =
          v + kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
          static_cast<size_t>(kv_head) * p.head_dim;
#pragma unroll
      for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int d = lane + item * GROUP_SIZE;
        if (d < p.head_dim) {
          acc[item] += beta * toFloat(v_ptr[d]);
        }
      }
    }
  }

  // 无效 Q 头用于保持线程块同步，但不会写回全局内存。
  if (active) {
    const float inv_sum =
        running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
      const int d = lane + item * GROUP_SIZE;
      if (d < p.head_dim) {
        o_ptr[d] = fromFloat<T>(acc[item] * inv_sum);
      }
    }
  }
}

/**
 * @brief 根据 head_dim 启动对应的 K/V 分块 FlashAttention Kernel
 * @tparam T 输入和输出张量的数据类型
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状、头数及因果掩码等参数
 */
template <typename T>
void launchKVTiledAttention(const T* d_q, const T* d_k, const T* d_v, T* d_o,
                            const AttnParams& params) {
  // float 的 head_dim=64 路径使用更小的 K/V tile，以控制共享内存占用。
  constexpr int HEAD64_KV_TILE =
      std::is_same<T, float>::value ? ATTENTION_FLOAT_HEAD64_KV_TILE_ROWS
                                    : ATTENTION_KV_TILE_ROWS;
#if defined(ATTENTION_USE_64_LANE_BACKEND)
  constexpr int HEAD32_GROUP_SIZE = 32;
  constexpr int HEAD32_QUERIES_PER_BLOCK = 16;
  constexpr int HEAD64_GROUP_SIZE = 64;
  constexpr int HEAD64_QUERIES_PER_BLOCK = 8;
#else
  constexpr int HEAD32_GROUP_SIZE = 32;
  constexpr int HEAD32_QUERIES_PER_BLOCK = 16;
  constexpr int HEAD64_GROUP_SIZE = 32;
  constexpr int HEAD64_QUERIES_PER_BLOCK = 16;
#endif
  // head_dim=32 和 64 使用各自匹配的线程组宽度、每 block 查询数及模板实例。
  if (params.head_dim == 32) {
    const int target_tiles =
        (params.target_seq_len + HEAD32_QUERIES_PER_BLOCK - 1) /
        HEAD32_QUERIES_PER_BLOCK;
    const int tile_blocks =
        params.batch_size * params.query_heads * target_tiles;
    flashAttentionKVTiledKernel<
        T, 32, HEAD32_GROUP_SIZE, HEAD32_QUERIES_PER_BLOCK,
        ATTENTION_KV_TILE_ROWS,
        ATTENTION_SCORE_TILE_ROWS>
        <<<tile_blocks, HEAD32_QUERIES_PER_BLOCK * HEAD32_GROUP_SIZE>>>(
            d_q, d_k, d_v, d_o, params);
  } else {
    const int target_tiles =
        (params.target_seq_len + HEAD64_QUERIES_PER_BLOCK - 1) /
        HEAD64_QUERIES_PER_BLOCK;
    const int tile_blocks =
        params.batch_size * params.query_heads * target_tiles;
    flashAttentionKVTiledKernel<
        T, 64, HEAD64_GROUP_SIZE, HEAD64_QUERIES_PER_BLOCK, HEAD64_KV_TILE,
        ATTENTION_SCORE_TILE_ROWS>
        <<<tile_blocks, HEAD64_QUERIES_PER_BLOCK * HEAD64_GROUP_SIZE>>>(
            d_q, d_k, d_v, d_o, params);
  }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  // 检查张量形状并准备 Host 输出空间。
  const size_t input_elems =
      checkedElementProduct(rows, hidden_dim, "rmsNorm input");
  if (h_input.size() != input_elems || h_weight.size() != hidden_dim) {
    throw std::invalid_argument("rmsNorm: input or weight shape mismatch");
  }
  h_output.resize(input_elems);
  if (input_elems == 0) {
    return;
  }

  // 网格 x 维直接使用 rows，因此先确保行数能够放入 dim3 的无符号整数范围。
  const RuntimeDeviceInfo& device = currentDeviceInfo();
  if (rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::overflow_error("rmsNorm: row count exceeds grid limit");
  }

  // 每个 Host 线程持有独立缓存，重复调用相同或更小形状时可以复用 GPU 显存。
  static thread_local ReusableDeviceBuffer<T> input_buffer;
  static thread_local ReusableDeviceBuffer<T> weight_buffer;
  static thread_local ReusableDeviceBuffer<T> output_buffer;

  // 复用或扩容三块设备缓冲区。
  input_buffer.ensure(input_elems, device.device_id);
  weight_buffer.ensure(hidden_dim, device.device_id);
  output_buffer.ensure(input_elems, device.device_id);

  T* d_input = input_buffer.data();
  T* d_weight = weight_buffer.data();
  T* d_output = output_buffer.data();

  // 将输入和权重从 Host 复制到当前 GPU。
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),
                           cudaMemcpyHostToDevice));

  // 选择不超过 256、设备上限和 hidden_dim 的最大 2 的幂，满足树形归约要求。
  int threads = 1;
  while (threads * 2 <= std::min(256, device.max_threads_per_block) &&
         static_cast<size_t>(threads * 2) <= hidden_dim) {
    threads *= 2;
  }
  // 一个 block 负责一行；动态共享内存为每个线程提供一个 FP32 归约槽位。
  dim3 blocks(rows);
  size_t shared_mem = threads * sizeof(float);
  rmsNormKernel<T><<<blocks, threads, shared_mem>>>(d_input, d_weight, d_output,
                                                    rows, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());

  // 默认流中的同步 D2H 拷贝会等待 Kernel 完成，因此无需额外设备同步。
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, input_elems * sizeof(T),
                           cudaMemcpyDeviceToHost));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  // 检查形状参数和 GQA 的查询头/KV 头关系。
  if (batch_size < 0 || target_seq_len < 0 || src_seq_len < 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument(
        "flashAttention: invalid non-positive shape parameter");
  }
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument(
        "flashAttention: query_heads must be divisible by kv_heads");
  }

  // 计算 Q、K/V 的元素数量，并检查乘法溢出。
  size_t q_elems = checkedElementProduct(
      checkedElementProduct(
          checkedElementProduct(static_cast<size_t>(batch_size),
                                static_cast<size_t>(target_seq_len), "query"),
          static_cast<size_t>(query_heads), "query"),
      static_cast<size_t>(head_dim), "query");
  size_t kv_elems = checkedElementProduct(
      checkedElementProduct(
          checkedElementProduct(static_cast<size_t>(batch_size),
                                static_cast<size_t>(src_seq_len), "key/value"),
          static_cast<size_t>(kv_heads), "key/value"),
      static_cast<size_t>(head_dim), "key/value");
  // 因果注意力中任何查询都不会访问 target_seq_len 之后的 K/V，可缩短设备端副本。
  const int device_src_seq_len =
      is_causal ? std::min(target_seq_len, src_seq_len) : src_seq_len;
  const size_t device_kv_elems = checkedElementProduct(
      checkedElementProduct(
          checkedElementProduct(static_cast<size_t>(batch_size),
                                static_cast<size_t>(device_src_seq_len),
                                "device key/value"),
          static_cast<size_t>(kv_heads), "device key/value"),
      static_cast<size_t>(head_dim), "device key/value");
  const size_t total_queries = q_elems / static_cast<size_t>(head_dim);
  if (total_queries > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("flashAttention: grid size overflow");
  }
  if (h_q.size() != q_elems || h_k.size() != kv_elems ||
      h_v.size() != kv_elems) {
    throw std::invalid_argument("flashAttention: input tensor shape mismatch");
  }
  // Host 输出会在末尾被 D2H 覆盖，因此禁止它与任一输入 vector 是同一对象。
  if (&h_o == &h_q || &h_o == &h_k || &h_o == &h_v) {
    throw std::invalid_argument("flashAttention: output must not alias input");
  }
  h_o.resize(q_elems);
  if (q_elems == 0 || kv_elems == 0) {
    std::fill(h_o.begin(), h_o.end(), fromFloat<T>(0.0f));
    return;
  }

  // 后续路径选择依赖 warp 宽度、线程上限和每 block 共享内存容量。
  const RuntimeDeviceInfo& device = currentDeviceInfo();

  static thread_local ReusableDeviceBuffer<T> q_buffer;
  static thread_local ReusableDeviceBuffer<T> k_buffer;
  static thread_local ReusableDeviceBuffer<T> v_buffer;
  static thread_local ReusableDeviceBuffer<T> o_buffer;

  // 复用或扩容 Q、K、V、O 的设备缓冲区。
  q_buffer.ensure(q_elems, device.device_id);
  k_buffer.ensure(device_kv_elems, device.device_id);
  v_buffer.ensure(device_kv_elems, device.device_id);
  o_buffer.ensure(q_elems, device.device_id);

  T* d_q = q_buffer.data();
  T* d_k = k_buffer.data();
  T* d_v = v_buffer.data();
  T* d_o = o_buffer.data();

  // 复制 Q；因果模式只复制实际可能访问的 K/V 前缀。
  RUNTIME_CHECK(
      cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), cudaMemcpyHostToDevice));
  if (device_src_seq_len == src_seq_len) {
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T),
                             cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T),
                             cudaMemcpyHostToDevice));
  } else {
    const size_t host_batch_pitch =
        static_cast<size_t>(src_seq_len) * kv_heads * head_dim * sizeof(T);
    const size_t device_batch_pitch = static_cast<size_t>(device_src_seq_len) *
                                      kv_heads * head_dim * sizeof(T);
    RUNTIME_CHECK(cudaMemcpy2D(d_k, device_batch_pitch, h_k.data(),
                               host_batch_pitch, device_batch_pitch,
                               batch_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy2D(d_v, device_batch_pitch, h_v.data(),
                               host_batch_pitch, device_batch_pitch,
                               batch_size, cudaMemcpyHostToDevice));
  }

  const int total = static_cast<int>(total_queries);
#if !defined(ATTENTION_USE_64_LANE_BACKEND)
  // 双 Kernel 路径需要保存 [total_queries, src_seq_len] 的 FP32 logits，限制为 256 MB。
  constexpr size_t CACHED_LOGIT_LIMIT_BYTES = 256ull << 20;
  constexpr size_t CACHED_LOGIT_LIMIT_ELEMENTS =
      CACHED_LOGIT_LIMIT_BYTES / sizeof(float);
  const bool cached_logits_affordable =
      device_src_seq_len > 0 &&
      total_queries <= CACHED_LOGIT_LIMIT_ELEMENTS /
                           static_cast<size_t>(device_src_seq_len);
  const size_t cached_logit_elems =
      cached_logits_affordable
          ? total_queries * static_cast<size_t>(device_src_seq_len)
          : 0;
#endif
  // 将原始形状和常用派生量打包，按值传给所有设备 Kernel。
  const AttnParams params{batch_size,
                          target_seq_len,
                          device_src_seq_len,
                          query_heads,
                          kv_heads,
                          head_dim,
                          total,
                          target_seq_len * query_heads,
                          query_heads / kv_heads,
                          is_causal};
  // 先根据问题规模判断分块是否有收益，再结合实际硬件资源决定能否启动。
  const bool kv_tiled_shape = std::is_same<T, float>::value
                                   ? shouldUseFloatKVTiled(
                                         total_queries, target_seq_len,
                                         device_src_seq_len, head_dim)
                                   : shouldUseHalfKVTiled(
                                         total_queries, target_seq_len,
                                         device_src_seq_len, head_dim);
#if defined(ATTENTION_USE_64_LANE_BACKEND)
  constexpr int EXPECTED_WARP_SIZE = 64;
#else
  constexpr int EXPECTED_WARP_SIZE = 32;
#endif
  const bool warp_width_matches = device.warp_size == EXPECTED_WARP_SIZE;
  size_t kv_tiled_shared_mem = 0;
  if (head_dim == 32) {
    kv_tiled_shared_mem = std::is_same<T, float>::value ? 18944u : 8192u;
  } else if (head_dim == 64) {
#if defined(ATTENTION_USE_64_LANE_BACKEND)
    kv_tiled_shared_mem = std::is_same<T, float>::value ? 18688u : 16384u;
#else
    kv_tiled_shared_mem = std::is_same<T, float>::value ? 20992u : 16384u;
#endif
  }
  // K/V 分块 Kernel 固定使用 512 线程，并要求其静态共享内存不超过设备上限。
  const bool use_kv_tiled =
      kv_tiled_shape && warp_width_matches &&
      device.max_threads_per_block >= 512 &&
      kv_tiled_shared_mem <= device.shared_mem_per_block;
  // 根据数据类型、形状和设备资源选择具体实现。
  if constexpr (std::is_same<T, float>::value) {
    constexpr int THREADS = 128;
    dim3 blocks(positiveCeilDiv(total, THREADS));
    // FP32 路径按优先级选择：超大维度共享内存路径、D32 查询分块、K/V 分块、
    // 缓存 logits 路径、常见固定维度特化，最后使用运行期容量档位 Kernel。
    constexpr int STACK_ACC_MAX = 1024;
    const size_t independent_tile_shared_mem =
        2ull * 64 * static_cast<size_t>(head_dim) * sizeof(float);
    const bool use_float_d32_query_tile =
        kv_tiled_shape && is_causal && head_dim == 32 &&
        device.max_threads_per_block >= 128 &&
        independent_tile_shared_mem <= device.shared_mem_per_block;
    // 优先使用复用程度更高的分块路径；头维度过大时改用共享内存或逐元素回退。
    if (use_float_d32_query_tile) {
      launchFloatD32QueryTiledAttention(d_q, d_k, d_v, d_o, params);
    } else if (use_kv_tiled) {
      launchKVTiledAttention<float>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim > STACK_ACC_MAX) {
      // 每个线程在共享内存中占用 head_dim 个 FP32；必要时减少 block 线程数。
      int acc_threads = THREADS;
      const size_t smem_per_thread = static_cast<size_t>(head_dim) * sizeof(float);
      while (acc_threads > 1 &&
             acc_threads * smem_per_thread > device.shared_mem_per_block) {
        acc_threads /= 2;
      }
      if (smem_per_thread > device.shared_mem_per_block) {
        launchElementwiseAttentionFallback(d_q, d_k, d_v, d_o, q_elems,
                                           params, device);
      } else {
        dim3 acc_blocks(positiveCeilDiv(total, acc_threads));
        const size_t shared_mem = acc_threads * smem_per_thread;
        flashAttentionFloatKernelSharedAcc
            <<<acc_blocks, acc_threads, shared_mem>>>(d_q, d_k, d_v, d_o,
                                                       params);
      }
#if !defined(ATTENTION_USE_64_LANE_BACKEND)
    } else if (!warp_width_matches) {
      launchElementwiseAttentionFallback(d_q, d_k, d_v, d_o, q_elems,
                                         params, device);
    } else if (head_dim == 32 && device_src_seq_len >= 128) {
      constexpr int WARPS_PER_BLOCK = THREADS / 32;
      dim3 warp_blocks(positiveCeilDiv(total, WARPS_PER_BLOCK));
      flashAttentionWarpKernelRegister<float, 32>
          <<<warp_blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
      // 缓存 logits 的输出 Kernel 依赖 32 线程 warp，因此只在匹配的平台启用。
    } else if ((head_dim == 64 || head_dim == 128 || head_dim == 256) &&
               cached_logits_affordable) {
      // 双 Kernel 缓存 logits，临时缓冲区限制为 256 MB；过大时改走重算路径。
      static thread_local ReusableDeviceBuffer<float> logits_buffer;
      static thread_local ReusableDeviceBuffer<float> rowmax_buffer;
      logits_buffer.ensure(cached_logit_elems, device.device_id);
      rowmax_buffer.ensure(total_queries, device.device_id);

      constexpr int WARPS_PER_BLOCK = THREADS / 32;
      dim3 blocks2(positiveCeilDiv(total, WARPS_PER_BLOCK));
      if (head_dim == 64 && device_src_seq_len >= 128) {
        flashAttentionWarpLogitsKernel<float, 64><<<blocks2, THREADS>>>(
            d_q, d_k, logits_buffer.data(), rowmax_buffer.data(), params);
      } else {
        dim3 blocks1(positiveCeilDiv(total, THREADS));
        flashAttentionFloatLogitsKernel<<<blocks1, THREADS>>>(
            d_q, d_k, logits_buffer.data(), rowmax_buffer.data(), params);
      }

      if (head_dim == 64) {
        flashAttentionFloatOutputKernel<float, 64><<<blocks2, THREADS>>>(
            logits_buffer.data(), rowmax_buffer.data(), d_v, d_o, params);
      } else if (head_dim == 128) {
        flashAttentionFloatOutputKernel<float, 128><<<blocks2, THREADS>>>(
            logits_buffer.data(), rowmax_buffer.data(), d_v, d_o, params);
      } else {
        flashAttentionFloatOutputKernel<float, 256><<<blocks2, THREADS>>>(
            logits_buffer.data(), rowmax_buffer.data(), d_v, d_o, params);
      }
#endif  // 非 64 线程 warp 后端
    // 常见的 2 的幂 head_dim 使用编译期定长特化，其余维度进入容量桶。
    } else if (head_dim == 1) {
      flashAttentionFloatKernelT<1>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 2) {
      flashAttentionFloatKernelT<2>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 4) {
      flashAttentionFloatKernelT<4>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 8) {
      flashAttentionFloatKernelT<8>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 16) {
      flashAttentionFloatKernelT<16>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 32) {
      flashAttentionFloatKernelT<32>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 64) {
      flashAttentionFloatKernelT<64>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim == 128) {
      flashAttentionFloatKernelT<128>
          <<<blocks, THREADS>>>(d_q, d_k, d_v, d_o, params);
    } else if (head_dim <= 128) {
      flashAttentionFloatKernel<128><<<blocks, THREADS>>>(d_q, d_k, d_v, d_o,
                                                          params);
    } else if (head_dim <= 256) {
      flashAttentionFloatKernel<256><<<blocks, THREADS>>>(d_q, d_k, d_v, d_o,
                                                          params);
    } else if (head_dim <= 512) {
      flashAttentionFloatKernel<512><<<blocks, THREADS>>>(d_q, d_k, d_v, d_o,
                                                          params);
    } else {
      flashAttentionFloatKernel<1024><<<blocks, THREADS>>>(d_q, d_k, d_v, d_o,
                                                           params);
    }
  } else {
#if defined(ATTENTION_USE_64_LANE_BACKEND)
    // 64-lane 后端优先使用原生线程组；资源不足时退回单线程查询或逐元素路径。
    constexpr int GROUP_SIZE = 64;
    constexpr int GROUPS_PER_BLOCK = 4;
    constexpr int SCORE_TILE = ATTENTION_SCORE_TILE_ROWS;
    constexpr int MAX_GROUP_HEAD_DIM = 1024;
    if (use_kv_tiled) {
      launchKVTiledAttention<T>(d_q, d_k, d_v, d_o, params);
    } else if (total_queries >= 16384 && head_dim <= MAX_GROUP_HEAD_DIM &&
        target_seq_len <= 65535 && batch_size <= 65535 &&
        device.max_threads_per_block >= GROUP_SIZE * GROUPS_PER_BLOCK) {
      const dim3 group_block(GROUP_SIZE * GROUPS_PER_BLOCK);
      const dim3 group_grid(
          positiveCeilDiv(query_heads, GROUPS_PER_BLOCK),
          target_seq_len, batch_size);

#define ATTENTION_LAUNCH_PORTABLE_GROUP(BUCKET)                           \
  flashAttentionPortableGroupKernel<T, GROUP_SIZE, GROUPS_PER_BLOCK,     \
                                    BUCKET, SCORE_TILE>                   \
      <<<group_grid, group_block>>>(d_q, d_k, d_v, d_o, params)

      if (head_dim <= GROUP_SIZE) {
        ATTENTION_LAUNCH_PORTABLE_GROUP(GROUP_SIZE);
      } else if (head_dim <= 2 * GROUP_SIZE) {
        ATTENTION_LAUNCH_PORTABLE_GROUP(2 * GROUP_SIZE);
      } else if (head_dim <= 4 * GROUP_SIZE) {
        ATTENTION_LAUNCH_PORTABLE_GROUP(4 * GROUP_SIZE);
      } else {
        ATTENTION_LAUNCH_PORTABLE_GROUP(MAX_GROUP_HEAD_DIM);
      }

#undef ATTENTION_LAUNCH_PORTABLE_GROUP
    } else {
      const size_t smem_per_thread =
          static_cast<size_t>(head_dim) * sizeof(float);
      int acc_threads = std::max(1, std::min(128, device.max_threads_per_block));
      while (acc_threads > 1 &&
             acc_threads * smem_per_thread > device.shared_mem_per_block) {
        acc_threads /= 2;
      }
      if (smem_per_thread > device.shared_mem_per_block) {
        launchElementwiseAttentionFallback(d_q, d_k, d_v, d_o, q_elems,
                                           params, device);
      } else {
        dim3 acc_blocks(positiveCeilDiv(total, acc_threads));
        const size_t shared_mem = acc_threads * smem_per_thread;
        flashAttentionThreadPerQueryKernel<T>
            <<<acc_blocks, acc_threads, shared_mem>>>(d_q, d_k, d_v, d_o,
                                                       params);
      }
    }
#else
    // NVIDIA half 路径保持线程数为 32 的倍数，使每个 warp 可以独立处理一个查询。
    const int preferred_threads =
        (device_src_seq_len >= 128 && total_queries >= 16384) ? 512 : 128;
    const int threads = std::max(
        32, (std::min(preferred_threads, device.max_threads_per_block) / 32) *
                32);
    const int warps_per_block = threads / 32;
    dim3 blocks(positiveCeilDiv(total, warps_per_block));

    // half 也可使用双 Kernel 缓存 logits；临时缓冲区过大时保留单遍在线路径。
    if (!warp_width_matches) {
      launchElementwiseAttentionFallback(d_q, d_k, d_v, d_o, q_elems,
                                         params, device);
    } else if (use_kv_tiled) {
      launchKVTiledAttention<T>(d_q, d_k, d_v, d_o, params);
    } else if ((head_dim == 128 || head_dim == 256) &&
               cached_logits_affordable) {
      static thread_local ReusableDeviceBuffer<float> h_logits_buffer;
      static thread_local ReusableDeviceBuffer<float> h_rowmax_buffer;
      h_logits_buffer.ensure(cached_logit_elems, device.device_id);
      h_rowmax_buffer.ensure(total_queries, device.device_id);

      if (head_dim == 128) {
        flashAttentionWarpLogitsKernel<T, 128><<<blocks, threads>>>(
            d_q, d_k, h_logits_buffer.data(), h_rowmax_buffer.data(), params);
        flashAttentionFloatOutputKernel<T, 128><<<blocks, threads>>>(
            h_logits_buffer.data(), h_rowmax_buffer.data(), d_v, d_o, params);
      } else {
        flashAttentionWarpLogitsKernel<T, 256><<<blocks, threads>>>(
            d_q, d_k, h_logits_buffer.data(), h_rowmax_buffer.data(), params);
        flashAttentionFloatOutputKernel<T, 256><<<blocks, threads>>>(
            h_logits_buffer.data(), h_rowmax_buffer.data(), d_v, d_o, params);
      }
    // 中小 head_dim 使用寄存器累加器，超大维度改用每 warp 一段共享内存。
    } else if (head_dim <= 64) {
      flashAttentionWarpKernelRegister<T, 64><<<blocks, threads>>>(
          d_q, d_k, d_v, d_o, params);
    } else if (head_dim <= 128) {
      flashAttentionWarpKernelRegister<T, 128><<<blocks, threads>>>(
          d_q, d_k, d_v, d_o, params);
    } else if (head_dim <= 256) {
      flashAttentionWarpKernelRegister<T, 256><<<blocks, threads>>>(
          d_q, d_k, d_v, d_o, params);
    } else {
      // 每个 warp 在共享内存中保存 Q 和累加器，并按设备容量减少每个 block 的 warp 数。
      int warps = warps_per_block;
      const size_t per_warp = 2ull * static_cast<size_t>(head_dim) * sizeof(float);
      while (warps > 1 &&
             warps * per_warp > device.shared_mem_per_block) {
        warps /= 2;
      }
      if (per_warp > device.shared_mem_per_block) {
        launchElementwiseAttentionFallback(d_q, d_k, d_v, d_o, q_elems,
                                           params, device);
      } else {
        const int acc_threads = warps * 32;
        dim3 acc_blocks(positiveCeilDiv(total, warps));
        const size_t shared_mem = warps * per_warp;
        flashAttentionWarpKernelShared<T>
            <<<acc_blocks, acc_threads, shared_mem>>>(d_q, d_k, d_v, d_o,
                                                       params);
      }
    }
#endif  // 64 线程 warp 后端
  }
  // 所有分支都只负责启动 Kernel，这里统一检查启动配置或异步提交错误。
  RUNTIME_CHECK(cudaGetLastError());

  // 默认流中的同步 D2H 拷贝会等待 Kernel 完成。
  RUNTIME_CHECK(
      cudaMemcpy(h_o.data(), d_o, q_elems * sizeof(T), cudaMemcpyDeviceToHost));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
