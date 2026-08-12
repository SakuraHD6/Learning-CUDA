#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <musa_fp16.h>

#include "../tester/utils.h"

/**
 * @brief 计算两个维度的乘积，并检查张量元素数量是否溢出
 * @param lhs 左操作数
 * @param rhs 右操作数
 * @param tensor_name 张量名称
 * @return 两个维度的乘积
 */
inline size_t checkedMusaElementProduct(size_t lhs, size_t rhs,
                                        const char* tensor_name) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(std::string(tensor_name) +
                              " element count overflow");
  }
  return lhs * rhs;
}

/**
 * @brief 计算正整数的向上取整除法
 * @param value 被除数
 * @param divisor 除数
 * @return 向上取整的商
 */
__host__ __device__ __forceinline__ int musaPositiveCeilDiv(int value,
                                                             int divisor) {
  return value / divisor + (value % divisor != 0);
}

/**
 * @brief 保存当前 MUSA 设备的运行时硬件信息
 */
struct MusaRuntimeDeviceInfo {
  int device_id;                 // 设备编号
  int warp_size;                 // 一个 warp 包含的线程数
  int multiprocessor_count;      // 多处理器数量
  int max_threads_per_block;     // 每个 block 的最大线程数
  int max_grid_x;                // grid.x 上限
  int max_grid_y;                // grid.y 上限
  int max_grid_z;                // grid.z 上限
  size_t shared_mem_per_block;   // 每个 block 的共享内存上限
};

/**
 * @brief 获取当前 MUSA 设备信息，并按 Host 线程缓存查询结果
 * @return 当前设备对应的运行时硬件信息
 */
inline const MusaRuntimeDeviceInfo& currentMusaDeviceInfo() {
  int current_device = 0;
  RUNTIME_CHECK(musaGetDevice(&current_device));
  // 每个 Host 线程维护独立缓存，切换设备后才重新读取属性。
  static thread_local MusaRuntimeDeviceInfo cached = {
      -1, 0, 0, 0, 0, 0, 0, 0};
  if (cached.device_id != current_device) {
    // 读取 warp、网格、线程和共享内存上限，供后续 Kernel 调度决策使用。
    musaDeviceProp properties = {};
    RUNTIME_CHECK(musaGetDeviceProperties(&properties, current_device));
    cached.device_id = current_device;
    cached.warp_size = properties.warpSize;
    cached.multiprocessor_count = properties.multiProcessorCount;
    cached.max_threads_per_block = properties.maxThreadsPerBlock;
    cached.max_grid_x = properties.maxGridSize[0];
    cached.max_grid_y = properties.maxGridSize[1];
    cached.max_grid_z = properties.maxGridSize[2];
    cached.shared_mem_per_block = properties.sharedMemPerBlock;
  }
  return cached;
}

// -----------------------------------------------------------------------------
// 调优常量
// -----------------------------------------------------------------------------

// 单个查询使用的协作线程数。归约通过共享内存完成，因此正确性不依赖 warp 宽度；
// 可以通过 -DMUSA_GROUP_SIZE=<width> 覆盖默认值。
#ifndef MUSA_GROUP_SIZE
#define MUSA_GROUP_SIZE 32
#endif

// 每个 block 处理的 Q 头组数；更大的值提高 GQA 中 K/V 复用，但增加共享内存开销。
#ifndef MUSA_GROUPS_PER_BLOCK
#define MUSA_GROUPS_PER_BLOCK 4
#endif

// 两次 block 屏障之间处理的 K/V 行数。
#ifndef MUSA_K_TILE
#define MUSA_K_TILE 8
#endif

static_assert(MUSA_GROUP_SIZE * MUSA_GROUPS_PER_BLOCK <= 1024,
              "block size exceeds the maximum threads per block");
static_assert(MUSA_GROUP_SIZE >= MUSA_K_TILE,
              "the score reduction assigns one lane per tile row");

constexpr size_t MUSA_KV_TILED_MIN_QUERIES = 16384;
constexpr int MUSA_HALF_KV_TILED_MIN_SEQUENCE = 64;
constexpr int MUSA_FLOAT_KV_TILED_MIN_SEQUENCE = 128;
constexpr int MUSA_KV_TILE_ROWS = 64;
constexpr int MUSA_HALF_HEAD64_KV_TILE_ROWS = 32;
constexpr int MUSA_FLOAT_HEAD64_KV_TILE_ROWS = 32;
constexpr int MUSA_SCORE_TILE_ROWS = 8;

// -----------------------------------------------------------------------------
// 标量类型转换：无论存储类型为何，Kernel 内部统一使用 FP32 累加。
// -----------------------------------------------------------------------------
/**
 * @brief 将类型 T 转换为 float，供 Kernel 使用 FP32 计算
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
 * @brief 将 float 类型转换为 half
 * @param x 待转换的 float 数据
 * @return 转换后的 half 数据
 */
template <>
__host__ __device__ __forceinline__ half fromFloat<half>(float x) {
  return __float2half(x);
}

// -----------------------------------------------------------------------------
// 可复用设备缓冲区：保留显存并只在容量不足或设备变化时重新分配。
// -----------------------------------------------------------------------------
/**
 * @brief 管理一块可复用且记录所属设备的 MUSA 显存
 * @tparam T 缓冲区元素类型
 */
template <typename T>
class ReusableDeviceBuffer {
 public:
  ReusableDeviceBuffer() = default;
  ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
  ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

  ~ReusableDeviceBuffer() {
    if (ptr_ != nullptr) {
      int restore_device = -1;
      if (musaGetDevice(&restore_device) == musaSuccess) {
        // musaFree 必须在内存所属设备执行，释放完成后恢复调用前设备。
        if (restore_device != device_id_) {
          musaSetDevice(device_id_);
        }
        musaFree(ptr_);
        if (restore_device != device_id_) {
          musaSetDevice(restore_device);
        }
      }
    }
  }

  /**
   * @brief 确保缓冲区位于指定设备且能够容纳所需元素
   * @param count 所需元素数量
   * @param current_device 当前 MUSA 设备编号
   */
  void ensure(size_t count, int current_device) {
    // 防止 count*sizeof(T) 溢出；容量足够且设备一致时直接复用。
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("MUSA device allocation size overflow");
    }
    if (device_id_ == current_device && count <= capacity_) {
      return;
    }
    // 设备变化或容量不足时，先切到原设备释放旧内存。
    if (ptr_ != nullptr) {
      if (device_id_ != current_device) {
        RUNTIME_CHECK(musaSetDevice(device_id_));
      }
      RUNTIME_CHECK(musaFree(ptr_));
      if (device_id_ != current_device) {
        RUNTIME_CHECK(musaSetDevice(current_device));
      }
      ptr_ = nullptr;
      capacity_ = 0;
      device_id_ = -1;
    }
    // musaMalloc 接收字节数；分配成功后记录容量和显存所属设备。
    RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    capacity_ = count;
    device_id_ = current_device;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t capacity() const { return capacity_; }

 private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
  int device_id_ = -1;
};

// -----------------------------------------------------------------------------
// RMSNorm
// -----------------------------------------------------------------------------

/**
 * @brief 使用一个 block 处理一行的 RMSNorm Kernel
 * @tparam T 输入、权重和输出的数据类型
 * @param input GPU 输入张量地址
 * @param weight GPU 权重向量地址
 * @param output GPU 输出张量地址
 * @param rows 输入行数
 * @param hidden_dim 每行元素数量
 * @param eps 防止除零的稳定项
 */
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output, size_t rows,
                              size_t hidden_dim, float eps) {
  // blockIdx.x 直接表示输入行号，一个 block 从头到尾只处理这一行。
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  // 每个线程计算部分平方和，并存入共享内存准备树形归约。
  extern __shared__ float shm[];
  float local_sum = 0.0f;

  // 线程以 blockDim.x 为步长遍历列，hidden_dim 大于线程数时也能覆盖整行。
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const float x = toFloat(input[row * hidden_dim + col]);
    local_sum += x * x;
  }

  shm[threadIdx.x] = local_sum;
  // 等待所有线程写入局部平方和，再开始读取其他线程的共享内存槽位。
  __syncthreads();

  // blockDim.x 由 Host 保证为 2 的幂，逐轮将后一半结果归约到前一半。
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shm[threadIdx.x] += shm[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // shm[0] 是整行平方和，除以 hidden_dim 得到均方值。
    const float mean_square = shm[0] / static_cast<float>(hidden_dim);
    // MUSA 数学头文件不保证提供 rsqrtf，因此显式计算 1/sqrt。
    shm[0] = 1.0f / sqrtf(mean_square + eps);
  }
  __syncthreads();

  // 所有线程读取同一个逆 RMS，并行完成归一化和逐列权重缩放。
  const float inv_rms = shm[0];
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const size_t idx = row * hidden_dim + col;
    const float x = toFloat(input[idx]);
    const float w = toFloat(weight[col]);
    output[idx] = fromFloat<T>(x * inv_rms * w);
  }
}

// -----------------------------------------------------------------------------
// FlashAttention
// -----------------------------------------------------------------------------

// “尚未读取分数”的哨兵值。使用有限大负数避免首次缩放出现 exp(-inf+inf)=NaN。
#define FA_NEG_SENTINEL (-1.0e30f)

/**
 * @brief 使用线程组和在线 Softmax 计算 MUSA FlashAttention
 * @tparam T 输入和输出张量的数据类型
 * @tparam GROUP_SIZE 每个查询使用的线程数
 * @tparam GROUPS_PER_BLOCK 每个 block 处理的查询组数
 * @tparam MAX_HEAD_DIM Kernel 支持的最大头维度
 * @tparam TILE 每轮处理的 K/V 行数
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param target_seq_len Q 的序列长度
 * @param src_seq_len K/V 的序列长度
 * @param query_heads Q 的注意力头数量
 * @param kv_heads K/V 的注意力头数量
 * @param head_dim 每个注意力头的维度
 * @param is_causal 是否使用因果掩码
 */
template <typename T, int GROUP_SIZE, int GROUPS_PER_BLOCK, int MAX_HEAD_DIM,
          int TILE>
__global__ void flashAttentionGroupKernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ o, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal) {
  constexpr int ITEMS_PER_LANE = (MAX_HEAD_DIM + GROUP_SIZE - 1) / GROUP_SIZE;

  // 保存 [线程组][tile 行][lane] 的局部点积；末维多一个元素以降低 bank 冲突。
  __shared__ float s_part[GROUPS_PER_BLOCK][TILE][GROUP_SIZE + 1];
  __shared__ float s_score[GROUPS_PER_BLOCK][TILE];

  // 三维网格映射 batch、目标位置和 Q 头组，组内 lane 分担 head_dim 分量。
  const int lane = threadIdx.x % GROUP_SIZE;
  const int group = threadIdx.x / GROUP_SIZE;

  const int b = blockIdx.z;
  const int t = blockIdx.y;
  const int qh = blockIdx.x * GROUPS_PER_BLOCK + group;
  const bool active = (qh < query_heads);
  // 越界线程组使用 Q 头 0 构造安全地址并继续同步，但最终不写回结果。
  const int qh_safe = active ? qh : 0;

  // GQA 中连续的多个 Q 头共享同一个 K/V 头。
  const int kv_head = qh_safe / (query_heads / kv_heads);

  const size_t q_base = (static_cast<size_t>(b) * target_seq_len * query_heads +
                         static_cast<size_t>(t) * query_heads + qh_safe) *
                        head_dim;
  const size_t kv_batch_base =
      static_cast<size_t>(b) * src_seq_len * kv_heads * head_dim;
  const size_t kv_row_stride = static_cast<size_t>(kv_heads) * head_dim;

  const T* q_ptr = q + q_base;
  T* o_ptr = o + q_base;

  // 每个 lane 将自己的 Q 分片和输出分片保存在寄存器中。
  float q_reg[ITEMS_PER_LANE];
  float acc[ITEMS_PER_LANE];

#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * GROUP_SIZE;
    q_reg[i] = (d < head_dim) ? toFloat(q_ptr[d]) : 0.0f;
    acc[i] = 0.0f;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float running_max = FA_NEG_SENTINEL;
  float running_sum = 0.0f;

  // 因果模式仅保留 s<=t。s_end 只依赖 blockIdx.y，因此 block 内循环次数一致。
  const int s_end =
      is_causal ? ((t + 1 < src_seq_len) ? (t + 1) : src_seq_len) : src_seq_len;

  for (int s0 = 0; s0 < s_end; s0 += TILE) {
    const int rows_here = ((s_end - s0) < TILE) ? (s_end - s0) : TILE;

    // 阶段 1：各 lane 计算当前 tile 每行 Q·K 的局部点积。
    for (int r = 0; r < rows_here; ++r) {
      const T* k_ptr =
          k + kv_batch_base +
          static_cast<size_t>(s0 + r) * kv_row_stride + kv_head * head_dim;
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int d = lane + i * GROUP_SIZE;
        if (d < head_dim) {
          dot += q_reg[i] * toFloat(k_ptr[d]);
        }
      }
      s_part[group][r][lane] = dot;
    }
    __syncthreads();

    // 阶段 2：前 rows_here 个 lane 分别归约一行，得到完整缩放分数。
    if (lane < rows_here) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < GROUP_SIZE; ++j) {
        sum += s_part[group][lane][j];
      }
      s_score[group][lane] = sum * scale;
    }
    __syncthreads();

    // 阶段 3：将当前 tile 合并进在线 Softmax，并同步缩放历史分母和输出。
    float tile_max = FA_NEG_SENTINEL;
    for (int r = 0; r < rows_here; ++r) {
      tile_max = fmaxf(tile_max, s_score[group][r]);
    }

    const float new_max = fmaxf(running_max, tile_max);
    const float alpha = expf(running_max - new_max);
    running_max = new_max;
    running_sum *= alpha;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
      acc[i] *= alpha;
    }

    for (int r = 0; r < rows_here; ++r) {
      const float beta = expf(s_score[group][r] - new_max);
      running_sum += beta;
      const T* v_ptr =
          v + kv_batch_base +
          static_cast<size_t>(s0 + r) * kv_row_stride + kv_head * head_dim;
#pragma unroll
      for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const int d = lane + i * GROUP_SIZE;
        if (d < head_dim) {
          acc[i] += beta * toFloat(v_ptr[d]);
        }
      }
    }
    // 此处无需额外屏障：下一轮会在重新读取共享分数前执行 __syncthreads()。
  }

  if (!active) {
    return;
  }

  // 有有效位置时最大分数贡献为 1，因此 running_sum 至少为 1；条件只保护退化情况。
  const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_LANE; ++i) {
    const int d = lane + i * GROUP_SIZE;
    if (d < head_dim) {
      o_ptr[d] = fromFloat<T>(acc[i] * inv_sum);
    }
  }
}

/**
 * @brief 汇总 MUSA FlashAttention 使用的张量形状和派生参数
 */
struct MusaAttnParams {
  int batch_size;           // 批次大小
  int target_seq_len;       // Q 的序列长度
  int src_seq_len;          // 设备端 K/V 的序列长度
  int query_heads;          // Q 的注意力头数量
  int kv_heads;             // K/V 的注意力头数量
  int head_dim;             // 每个注意力头的维度
  int query_heads_per_kv;   // 每个 K/V 头对应的 Q 头数量
  bool is_causal;           // 是否使用因果掩码
};

/**
 * @brief 使用运行期头维度执行严格两遍 FP32 FlashAttention
 * @tparam ACC_CAP 每个线程的最大输出累加器容量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 */
template <int ACC_CAP>
__global__ void flashAttentionMusaFloatStrictKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, MusaAttnParams p) {
  // 一个线程处理一个查询，并从一维索引解码 batch、目标位置和 Q/KV 头。
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int queries_per_batch = p.target_seq_len * p.query_heads;
  const int total_queries = p.batch_size * queries_per_batch;
  if (query_index >= total_queries) {
    return;
  }

  // 展平顺序为 [batch, target_position, query_head]，逐层还原 b、t、qh。
  const int b = query_index / queries_per_batch;
  const int rem = query_index - b * queries_per_batch;
  const int t = rem / p.query_heads;
  const int qh = rem - t * p.query_heads;
  // GQA 中连续的多个 Q 头共享同一个 K/V 头。
  const int kv_head = qh / p.query_heads_per_kv;

  // Q/O 布局为 [B,T,QH,D]，K/V 布局为 [B,S,KVH,D]。
  const size_t q_base = static_cast<size_t>(query_index) * p.head_dim;
  const size_t kv_batch_base =
      static_cast<size_t>(b) * p.src_seq_len * p.kv_heads * p.head_dim;
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * p.head_dim;
  // 非因果模式读取完整源序列；因果模式只读取当前目标位置及其之前的 K/V。
  const int s_end =
      p.is_causal ? (((t + 1) < p.src_seq_len) ? (t + 1) : p.src_seq_len)
                  : p.src_seq_len;
  const float scale = rsqrtf(static_cast<float>(p.head_dim));
  const float* q_ptr = q + q_base;

  // 第一遍计算最大缩放分数，第二遍计算分母和未归一化的 weight·V。
  float max_logit = FA_NEG_SENTINEL;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * p.head_dim;
    const float* k_ptr = k + kv_base;
    float dot = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    max_logit = fmaxf(max_logit, dot * scale);
  }

  float acc[ACC_CAP];
  for (int d = 0; d < p.head_dim; ++d) {
    acc[d] = 0.0f;
  }

  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * p.head_dim;
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;
    float dot = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const float weight = expf(dot * scale - max_logit);
    denom += weight;
    for (int d = 0; d < p.head_dim; ++d) {
      acc[d] += weight * v_ptr[d];
    }
  }

  // acc 是未归一化的 V 加权和，最后统一除以 Softmax 分母。
  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < p.head_dim; ++d) {
    o[q_base + d] = acc[d] * inv_denom;
  }
}

/**
 * @brief 使用编译期固定头维度执行严格两遍 FP32 FlashAttention
 * @tparam HEAD_DIM 每个注意力头的维度
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 */
template <int HEAD_DIM>
__global__ void flashAttentionMusaFloatStrictFixedKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, MusaAttnParams p) {
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int queries_per_batch = p.target_seq_len * p.query_heads;
  const int total_queries = p.batch_size * queries_per_batch;
  if (query_index >= total_queries) {
    return;
  }

  // HEAD_DIM 为编译期常量，坐标解码不变，但维度循环可以完全展开。
  const int b = query_index / queries_per_batch;
  const int rem = query_index - b * queries_per_batch;
  const int t = rem / p.query_heads;
  const int qh = rem - t * p.query_heads;
  const int kv_head = qh / p.query_heads_per_kv;
  const size_t q_base = static_cast<size_t>(query_index) * HEAD_DIM;
  const size_t kv_batch_base =
      static_cast<size_t>(b) * p.src_seq_len * p.kv_heads * HEAD_DIM;
  const size_t kv_row_stride = static_cast<size_t>(p.kv_heads) * HEAD_DIM;
  const int s_end =
      p.is_causal ? (((t + 1) < p.src_seq_len) ? (t + 1) : p.src_seq_len)
                  : p.src_seq_len;
  const float scale = rsqrtf(static_cast<float>(p.head_dim));
  const float* q_ptr = q + q_base;

  // 第一遍只求整行最大分数，第二遍重新计算分数并累加分母和 V。
  float max_logit = FA_NEG_SENTINEL;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * HEAD_DIM;
    const float* k_ptr = k + kv_base;
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    max_logit = fmaxf(max_logit, dot * scale);
  }

  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * HEAD_DIM;
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const float weight = expf(dot * scale - max_logit);
    denom += weight;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      acc[d] += weight * v_ptr[d];
    }
  }

  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    o[q_base + d] = acc[d] * inv_denom;
  }
}

/**
 * @brief 按参考实现的运算顺序计算 FP32 FlashAttention
 * @tparam ACC_CAP 每个线程的最大输出累加器容量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 */
template <int ACC_CAP>
__global__ void flashAttentionMusaFloatReferenceOrderKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, MusaAttnParams p) {
  const int query_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int queries_per_batch = p.target_seq_len * p.query_heads;
  const int total_queries = p.batch_size * queries_per_batch;
  if (query_index >= total_queries) {
    return;
  }

  // 此路径保持参考实现的遍历和舍入顺序，用于因果 D32 等数值敏感形状。
  const int b = query_index / queries_per_batch;
  const int rem = query_index - b * queries_per_batch;
  const int t = rem / p.query_heads;
  const int qh = rem - t * p.query_heads;
  const int kv_head = qh / p.query_heads_per_kv;
  const size_t q_base = static_cast<size_t>(query_index) * p.head_dim;
  const size_t kv_batch_base =
      static_cast<size_t>(b) * p.src_seq_len * p.kv_heads * p.head_dim;
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * p.head_dim;
  const int s_end =
      p.is_causal ? (((t + 1) < p.src_seq_len) ? (t + 1) : p.src_seq_len)
                  : p.src_seq_len;
  const float scale = rsqrtf(static_cast<float>(p.head_dim));
  const float* q_ptr = q + q_base;

  float max_logit = FA_NEG_SENTINEL;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * p.head_dim;
    const float* k_ptr = k + kv_base;
    float dot = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_ptr[d], k_ptr[d]));
    }
    max_logit = fmaxf(max_logit, dot * scale);
  }

  float denom = 0.0f;
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * p.head_dim;
    const float* k_ptr = k + kv_base;
    float dot = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_ptr[d], k_ptr[d]));
    }
    denom = __fadd_rn(denom, expf(dot * scale - max_logit));
  }

  float acc[ACC_CAP];
  for (int d = 0; d < p.head_dim; ++d) {
    acc[d] = 0.0f;
  }
  for (int s = 0; s < s_end; ++s) {
    const size_t kv_base =
        kv_batch_base + static_cast<size_t>(s) * kv_row_stride +
        static_cast<size_t>(kv_head) * p.head_dim;
    const float* k_ptr = k + kv_base;
    const float* v_ptr = v + kv_base;
    float dot = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      dot = __fadd_rn(dot, __fmul_rn(q_ptr[d], k_ptr[d]));
    }
    const float weight = expf(dot * scale - max_logit);
    for (int d = 0; d < p.head_dim; ++d) {
      acc[d] = __fadd_rn(acc[d], __fmul_rn(weight, v_ptr[d]));
    }
  }

  const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
  for (int d = 0; d < p.head_dim; ++d) {
    o[q_base + d] = acc[d] * inv_denom;
  }
}

/**
 * @brief 保存 MUSA K/V 分块 Kernel 中一个查询的坐标、边界和地址偏移
 */
struct MusaKVTiledCoord {
  int b;                    // 当前 batch 编号
  int qh;                   // 当前 Q 头编号
  int t0;                   // 当前查询 tile 的起始位置
  int t;                    // 当前查询位置
  int kv_head;              // 对应的 K/V 头编号
  int s_end;                // 当前查询允许访问的 K/V 行数
  int block_s_end;          // 整个 block 需要加载的 K/V 行数
  size_t q_base;            // 当前 Q/O 行首偏移
  size_t kv_batch_base;     // 当前 batch 的 K/V 起始偏移
  bool active;              // 当前查询是否有效
};

/**
 * @brief 解码 MUSA K/V 分块任务并计算查询坐标和访问边界
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam QUERIES_PER_BLOCK 每个 block 处理的查询数量
 * @param block_index 当前 block 的一维索引
 * @param query_in_block 当前查询在 block 内的编号
 * @param p 注意力形状和掩码参数
 * @return 当前分块查询的坐标与边界信息
 */
template <int HEAD_DIM, int QUERIES_PER_BLOCK>
__device__ __forceinline__ MusaKVTiledCoord decodeMusaKVTiledQuery(
    int block_index, int query_in_block, const MusaAttnParams& p) {
  MusaKVTiledCoord c;
  // 将 block_index 按 [batch, Q 头, target tile] 的布局逐层解码。
  const int target_tiles =
      musaPositiveCeilDiv(p.target_seq_len, QUERIES_PER_BLOCK);
  const int tiles_per_batch = p.query_heads * target_tiles;
  c.b = block_index / tiles_per_batch;
  const int block_rem = block_index - c.b * tiles_per_batch;
  c.qh = block_rem / target_tiles;
  const int target_tile = block_rem - c.qh * target_tiles;
  // t0 是查询 tile 的起始位置，query_in_block 决定当前线程或线程组负责的 t。
  c.t0 = target_tile * QUERIES_PER_BLOCK;
  c.t = c.t0 + query_in_block;
  c.active = c.t < p.target_seq_len;
  // 最后一个 tile 可能不满，无效查询使用 safe_t=0 构造安全地址但不参与计算。
  const int safe_t = c.active ? c.t : 0;
  // 将 Q 头映射到共享 K/V 头，并按 [B,T,QH,D] 计算 Q/O 行首偏移。
  c.kv_head = c.qh / p.query_heads_per_kv;
  c.q_base =
      (static_cast<size_t>(c.b) * p.target_seq_len * p.query_heads +
       static_cast<size_t>(safe_t) * p.query_heads + c.qh) *
      HEAD_DIM;
  c.kv_batch_base =
      static_cast<size_t>(c.b) * p.src_seq_len * p.kv_heads * HEAD_DIM;
  // s_end 是单查询因果边界，block_s_end 是整个查询 tile 的最大加载边界。
  c.s_end = c.active
                ? (p.is_causal
                       ? (((c.t + 1) < p.src_seq_len) ? (c.t + 1)
                                                       : p.src_seq_len)
                       : p.src_seq_len)
                : 0;
  // 整个 block 最多需要加载到查询 tile 末尾，同时不能超过源序列长度。
  const int block_target_end =
      ((c.t0 + QUERIES_PER_BLOCK) < p.target_seq_len)
          ? (c.t0 + QUERIES_PER_BLOCK)
          : p.target_seq_len;
  c.block_s_end =
      p.is_causal
          ? ((block_target_end < p.src_seq_len) ? block_target_end
                                                 : p.src_seq_len)
          : p.src_seq_len;
  return c;
}

/**
 * @brief 使用查询分块和固定最大值计算因果 FP32 D32 FlashAttention
 * @tparam HEAD_DIM 每个注意力头的维度，当前特化为 32
 * @tparam QUERIES_PER_BLOCK 每个 block 处理的查询数量
 * @tparam KV_TILE 每次载入共享内存的 K/V 行数
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 *
 * 一个线程负责一个查询，整个 block 协作加载 64 行 K/V。共享内存固定占用
 * 2 * 64 * 32 * sizeof(float) = 16 KB，可在 mp_21 的 28 KB 限制内运行。
 */
template <int HEAD_DIM, int QUERIES_PER_BLOCK, int KV_TILE>
__global__ void flashAttentionMusaFloatD32QueryTiledKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, MusaAttnParams p) {
  static_assert(HEAD_DIM == 32, "this kernel is specialized for D32");

  // block 中所有查询复用相同的 K/V tile，每个线程在寄存器中保存完整 Q 和输出。
  __shared__ float k_tile[KV_TILE * HEAD_DIM];
  __shared__ float v_tile[KV_TILE * HEAD_DIM];

  // 每个线程拥有一个查询；blockIdx.x 解码出 batch、Q 头和查询 tile。
  const MusaKVTiledCoord c =
      decodeMusaKVTiledQuery<HEAD_DIM, QUERIES_PER_BLOCK>(
          blockIdx.x, threadIdx.x, p);

  // 完整 Q 和输出累加器放在当前线程寄存器中，K/V tile 放在共享内存中复用。
  float q_values[HEAD_DIM];
  float output[HEAD_DIM];
  if (c.active) {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      q_values[d] = q[c.q_base + d];
    }
  }
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) {
    output[d] = 0.0f;
  }

  float max_logit = FA_NEG_SENTINEL;
  const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * HEAD_DIM;

  // 第一遍：求整行固定最大值，避免反复合并独立归一化 tile 带来的舍入漂移。
  for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
    const int rows = ((c.block_s_end - s0) < KV_TILE)
                         ? (c.block_s_end - s0)
                         : KV_TILE;
    // block 内线程协作加载 K；第一遍只求最大值，因此不需要载入 V。
    for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
      const int row = idx / HEAD_DIM;
      const int d = idx - row * HEAD_DIM;
      const size_t source =
          c.kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
          static_cast<size_t>(c.kv_head) * HEAD_DIM + d;
      k_tile[idx] = k[source];
    }
    __syncthreads();

    // 无效查询仍参加 block 同步，但不会读取 Q 或更新自己的最大值。
    if (c.active) {
      const int remaining = c.s_end - s0;
      const int valid_rows =
          remaining > 0 ? ((remaining < rows) ? remaining : rows) : 0;
      for (int row = 0; row < valid_rows; ++row) {
        float score = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          score = fmaf(q_values[d], k_tile[row * HEAD_DIM + d], score);
        }
        max_logit = fmaxf(max_logit, score * scale);
      }
    }
    __syncthreads();
  }

  // 第二遍：按 64 行 tile 累加未归一化概率与 V，整行结束后只归一化一次。
  float denom = 0.0f;
  for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
    const int rows = ((c.block_s_end - s0) < KV_TILE)
                         ? (c.block_s_end - s0)
                         : KV_TILE;
    // 第二遍同时加载 K 和 V，重算分数后立即累加对应的 V 行。
    for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
      const int row = idx / HEAD_DIM;
      const int d = idx - row * HEAD_DIM;
      const size_t source =
          c.kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
          static_cast<size_t>(c.kv_head) * HEAD_DIM + d;
      k_tile[idx] = k[source];
      v_tile[idx] = v[source];
    }
    __syncthreads();

    if (c.active) {
      const int remaining = c.s_end - s0;
      const int valid_rows =
          remaining > 0 ? ((remaining < rows) ? remaining : rows) : 0;
      // 先形成当前 tile 的局部分母和局部输出，再合并到整行累加状态。
      float tile_denom = 0.0f;
      float tile_output[HEAD_DIM];
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) {
        tile_output[d] = 0.0f;
      }
      for (int row = 0; row < valid_rows; ++row) {
        float score = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          score = fmaf(q_values[d], k_tile[row * HEAD_DIM + d], score);
        }
        const float weight = expf(score * scale - max_logit);
        tile_denom += weight;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          tile_output[d] =
              fmaf(weight, v_tile[row * HEAD_DIM + d], tile_output[d]);
        }
      }
      denom += tile_denom;
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) {
        output[d] += tile_output[d];
      }
    }
    __syncthreads();
  }

  // 所有 tile 处理完后只除一次整行分母，减少重复归一化带来的舍入误差。
  if (c.active) {
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
      o[c.q_base + d] = output[d] * inv_denom;
    }
  }
}

/**
 * @brief 在形状和设备资源满足条件时启动 FP32 D32 查询分块 Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状和掩码参数
 * @param total_queries 查询总数
 * @param device 当前 MUSA 设备信息
 * @return 已启动专用 Kernel 时返回 true，否则返回 false
 */
inline bool launchMusaFloatD32QueryTiledIfEligible(
    const float* d_q, const float* d_k, const float* d_v, float* d_o,
    const MusaAttnParams& params, size_t total_queries,
    const MusaRuntimeDeviceInfo& device) {
  constexpr int HEAD_DIM = 32;
  constexpr int QUERIES_PER_BLOCK = 128;
  constexpr int KV_TILE = 64;
  // 只为大规模因果 FP32 D32 启用，并要求设备支持 128 线程和 16 KB 共享内存。
  const bool eligible =
      params.is_causal && params.head_dim == HEAD_DIM &&
      total_queries >= MUSA_KV_TILED_MIN_QUERIES &&
      params.target_seq_len >= MUSA_HALF_KV_TILED_MIN_SEQUENCE &&
      params.src_seq_len >= MUSA_HALF_KV_TILED_MIN_SEQUENCE &&
      device.max_threads_per_block >= QUERIES_PER_BLOCK &&
      2u * KV_TILE * HEAD_DIM * sizeof(float) <=
          device.shared_mem_per_block;
  // 条件不满足时不启动任何 Kernel，由上层继续选择通用分块或回退路径。
  if (!eligible) {
    return false;
  }

  // 网格中每个 block 对应一个 batch、一个 Q 头和一个 128 查询 tile。
  const int target_tiles =
      musaPositiveCeilDiv(params.target_seq_len, QUERIES_PER_BLOCK);
  const int blocks = params.batch_size * params.query_heads * target_tiles;
  flashAttentionMusaFloatD32QueryTiledKernel<HEAD_DIM, QUERIES_PER_BLOCK,
                                             KV_TILE>
      <<<blocks, QUERIES_PER_BLOCK>>>(d_q, d_k, d_v, d_o, params);
  return true;
}

/**
 * @brief half 类型不使用 FP32 D32 查询分块路径
 * @return 始终返回 false
 */
inline bool launchMusaFloatD32QueryTiledIfEligible(
    const half*, const half*, const half*, half*, const MusaAttnParams&,
    size_t, const MusaRuntimeDeviceInfo&) {
  return false;
}

/**
 * @brief 对 K/V 分块并使用严格三遍扫描计算 FP32 FlashAttention
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam GROUP_SIZE 每个查询使用的线程数
 * @tparam QUERIES_PER_BLOCK 每个 block 处理的查询数量
 * @tparam KV_TILE 每次载入共享内存的 K/V 行数
 * @tparam SCORE_TILE 每组缓存的注意力分数数量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 */
template <int HEAD_DIM, int GROUP_SIZE, int QUERIES_PER_BLOCK, int KV_TILE,
          int SCORE_TILE>
__global__ void flashAttentionMusaFloatKVTiledKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ o, MusaAttnParams p) {
  static_assert(HEAD_DIM <= GROUP_SIZE,
                "float tiled kernel assigns at most one output per lane");

  // 所有查询组共享 K/V tile；Q 和小块分数也放入共享内存供顺序计算。
  __shared__ float k_tile[KV_TILE * HEAD_DIM];
  __shared__ float v_tile[KV_TILE * HEAD_DIM];
  __shared__ float q_shared[QUERIES_PER_BLOCK][HEAD_DIM];
  __shared__ float scores[QUERIES_PER_BLOCK][SCORE_TILE];

  const int lane = threadIdx.x % GROUP_SIZE;
  const int group = threadIdx.x / GROUP_SIZE;
  // 一个线程组对应一个查询；lane 0..HEAD_DIM-1 各负责一个输出维度。
  const MusaKVTiledCoord c =
      decodeMusaKVTiledQuery<HEAD_DIM, QUERIES_PER_BLOCK>(blockIdx.x, group, p);
  // 将各 lane 的 Q 分量拼成共享内存中的连续行，后续每个分数都可以复用。
  if (lane < HEAD_DIM) {
    q_shared[group][lane] = c.active ? q[c.q_base + lane] : 0.0f;
  }
  __syncthreads();

  float max_logit = FA_NEG_SENTINEL;
  float denom = 0.0f;
  float acc = 0.0f;
  const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * HEAD_DIM;

  // 三遍分别求最大值、Softmax 分母和 V 加权和，保持严格的数值顺序。
  for (int pass = 0; pass < 3; ++pass) {
    for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
      const int rows = ((c.block_s_end - s0) < KV_TILE)
                           ? (c.block_s_end - s0)
                           : KV_TILE;
      // 整个 block 协作加载 K；只有输出遍 pass=2 需要同时加载 V。
      for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
        const int row = idx / HEAD_DIM;
        const int d = idx - row * HEAD_DIM;
        const size_t source =
            c.kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
            static_cast<size_t>(c.kv_head) * HEAD_DIM + d;
        k_tile[idx] = k[source];
        if (pass == 2) {
          v_tile[idx] = v[source];
        }
      }
      __syncthreads();

      // query_rows 是当前查询在本 K/V tile 中真正可见的行数。
      const int remaining = c.s_end - s0;
      const int query_rows =
          remaining > 0 ? ((remaining < rows) ? remaining : rows) : 0;
      for (int row0 = 0; row0 < rows; row0 += SCORE_TILE) {
        const int score_rows =
            ((rows - row0) < SCORE_TILE) ? (rows - row0) : SCORE_TILE;
        // 前 score_rows 个 lane 各自计算一行完整 Q·K，并写入小块分数缓存。
        if (lane < score_rows) {
          // Q·K 使用分离的乘法与加法，避免 MUSA 融合 FMA 的舍入行为偏离参考结果。
          float dot = 0.0f;
#pragma unroll
          for (int d = 0; d < HEAD_DIM; ++d) {
            dot += q_shared[group][d] * k_tile[(row0 + lane) * HEAD_DIM + d];
          }
          scores[group][lane] = dot * scale;
        }
        __syncthreads();

        const int valid_remaining = query_rows - row0;
        const int valid_rows =
            valid_remaining > 0
                ? ((valid_remaining < score_rows) ? valid_remaining
                                                   : score_rows)
                : 0;
        // pass=0 求 max；pass=1 求 Σexp(score-max)；pass=2 求 Σweight·V。
        if (pass == 0) {
          for (int row = 0; row < valid_rows; ++row) {
            max_logit = fmaxf(max_logit, scores[group][row]);
          }
        } else if (pass == 1) {
          for (int row = 0; row < valid_rows; ++row) {
            denom += expf(scores[group][row] - max_logit);
          }
        } else {
          for (int row = 0; row < valid_rows; ++row) {
            const float weight = expf(scores[group][row] - max_logit);
            if (lane < HEAD_DIM) {
              // V 累加同样使用 mul+add，保持在 FP32 参考误差范围内。
              acc += weight * v_tile[(row0 + row) * HEAD_DIM + lane];
            }
          }
        }
        __syncthreads();
      }
      __syncthreads();
    }
  }

  // 仅有效查询和有效维度执行最终归一化及全局内存写回。
  if (c.active && lane < HEAD_DIM) {
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    o[c.q_base + lane] = acc * inv_denom;
  }
}

/**
 * @brief 对 K/V 分块并使用在线 Softmax 计算 half FlashAttention
 * @tparam T 输入和输出张量的数据类型
 * @tparam HEAD_DIM 每个注意力头的维度
 * @tparam GROUP_SIZE 每个查询使用的线程数
 * @tparam QUERIES_PER_BLOCK 每个 block 处理的查询数量
 * @tparam KV_TILE 每次载入共享内存的 K/V 行数
 * @tparam SCORE_TILE 每组缓存的注意力分数数量
 * @param q GPU 查询张量地址
 * @param k GPU 键张量地址
 * @param v GPU 值张量地址
 * @param o GPU 输出张量地址
 * @param p 注意力形状和掩码参数
 */
template <typename T, int HEAD_DIM, int GROUP_SIZE, int QUERIES_PER_BLOCK,
          int KV_TILE, int SCORE_TILE>
__global__ void flashAttentionMusaHalfKVTiledKernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ o, MusaAttnParams p) {
  static_assert(HEAD_DIM <= GROUP_SIZE,
                "half tiled kernel assigns at most one output per lane");

  // K/V tile 由整个 block 共享，partial 和 scores 用于线程组内点积归约。
  __shared__ T k_tile[KV_TILE * HEAD_DIM];
  __shared__ T v_tile[KV_TILE * HEAD_DIM];
  __shared__ float partial[QUERIES_PER_BLOCK][SCORE_TILE][GROUP_SIZE + 1];
  __shared__ float scores[QUERIES_PER_BLOCK][SCORE_TILE];

  const int lane = threadIdx.x % GROUP_SIZE;
  const int group = threadIdx.x / GROUP_SIZE;
  // 每组处理一个查询，每个有效 lane 保存一个 Q 分量和对应的输出累加标量。
  const MusaKVTiledCoord c =
      decodeMusaKVTiledQuery<HEAD_DIM, QUERIES_PER_BLOCK>(blockIdx.x, group, p);
  const float q_value =
      (c.active && lane < HEAD_DIM) ? toFloat(q[c.q_base + lane]) : 0.0f;
  float acc = 0.0f;
  float running_max = FA_NEG_SENTINEL;
  float running_sum = 0.0f;
  const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  const size_t kv_row_stride =
      static_cast<size_t>(p.kv_heads) * HEAD_DIM;

  // 沿源序列推进 K/V tile，并将每个分数小块合并进在线 Softmax 状态。
  for (int s0 = 0; s0 < c.block_s_end; s0 += KV_TILE) {
    const int rows = ((c.block_s_end - s0) < KV_TILE)
                         ? (c.block_s_end - s0)
                         : KV_TILE;
    // block 内所有线程协作把当前 K/V tile 搬入共享内存。
    for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += blockDim.x) {
      const int row = idx / HEAD_DIM;
      const int d = idx - row * HEAD_DIM;
      const size_t source =
          c.kv_batch_base + static_cast<size_t>(s0 + row) * kv_row_stride +
          static_cast<size_t>(c.kv_head) * HEAD_DIM + d;
      k_tile[idx] = k[source];
      v_tile[idx] = v[source];
    }
    __syncthreads();

    // query_rows 排除因果边界之后的 K/V 行，剩余共享数据仍可供其他查询使用。
    const int remaining = c.s_end - s0;
    const int query_rows =
        remaining > 0 ? ((remaining < rows) ? remaining : rows) : 0;
    for (int row0 = 0; row0 < rows; row0 += SCORE_TILE) {
      const int score_rows =
          ((rows - row0) < SCORE_TILE) ? (rows - row0) : SCORE_TILE;
      // 每个 lane 先计算 Q[d]·K[row,d]，形成该行点积的局部贡献。
      for (int row = 0; row < score_rows; ++row) {
        const float dot = lane < HEAD_DIM
                              ? q_value *
                                    toFloat(k_tile[(row0 + row) * HEAD_DIM + lane])
                              : 0.0f;
        partial[group][row][lane] = dot;
      }
      __syncthreads();

      // 前 score_rows 个 lane 分别归约一行的所有维度，并乘 1/sqrt(head_dim)。
      if (lane < score_rows) {
        float dot = 0.0f;
#pragma unroll
        for (int source_lane = 0; source_lane < GROUP_SIZE; ++source_lane) {
          dot += partial[group][lane][source_lane];
        }
        scores[group][lane] = dot * scale;
      }
      __syncthreads();

      const int valid_remaining = query_rows - row0;
      const int valid_rows =
          valid_remaining > 0
              ? ((valid_remaining < score_rows) ? valid_remaining : score_rows)
              : 0;
      float tile_max = FA_NEG_SENTINEL;
      for (int row = 0; row < valid_rows; ++row) {
        tile_max = fmaxf(tile_max, scores[group][row]);
      }
      // 使用新最大值重缩放历史分母和输出，再加入当前分数小块。
      const float new_max = fmaxf(running_max, tile_max);
      const float alpha = expf(running_max - new_max);
      running_max = new_max;
      running_sum *= alpha;
      acc *= alpha;

      // beta 是当前行在新基准下的权重，各 lane 用它更新对应 V 维度。
      for (int row = 0; row < valid_rows; ++row) {
        const float beta = expf(scores[group][row] - new_max);
        running_sum += beta;
        if (lane < HEAD_DIM) {
          acc += beta * toFloat(v_tile[(row0 + row) * HEAD_DIM + lane]);
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }

  // 在线累加结果仍未归一化，最后除以 running_sum 并转换回输出类型。
  if (c.active && lane < HEAD_DIM) {
    const float inv_sum = running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
    o[c.q_base + lane] = fromFloat<T>(acc * inv_sum);
  }
}

/**
 * @brief 判断当前形状是否适合使用 MUSA K/V 分块路径
 * @param total_queries 查询总数
 * @param target_seq_len Q 的序列长度
 * @param src_seq_len K/V 的序列长度
 * @param head_dim 每个注意力头的维度
 * @param min_sequence 启用分块路径所需的最小序列长度
 * @return 满足分块条件时返回 true，否则返回 false
 */
inline bool shouldUseMusaKVTiled(size_t total_queries, int target_seq_len,
                                 int src_seq_len, int head_dim,
                                 int min_sequence) {
  return total_queries >= MUSA_KV_TILED_MIN_QUERIES &&
         target_seq_len >= min_sequence && src_seq_len >= min_sequence &&
         (head_dim == 32 || head_dim == 64);
}

/**
 * @brief 根据 head_dim 启动 FP32 MUSA K/V 分块 Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状和掩码参数
 */
inline void launchMusaKVTiled(const float* d_q, const float* d_k,
                              const float* d_v, float* d_o,
                              const MusaAttnParams& params) {
  // D32 使用 32 线程组、每 block 16 个查询；D64 使用 64 线程组、每 block 8 个查询。
  if (params.head_dim == 32) {
    constexpr int GROUP_SIZE = 32;
    constexpr int QUERIES_PER_BLOCK = 16;
    const int target_tiles =
        musaPositiveCeilDiv(params.target_seq_len, QUERIES_PER_BLOCK);
    const int blocks = params.batch_size * params.query_heads * target_tiles;
    flashAttentionMusaFloatKVTiledKernel<
        32, GROUP_SIZE, QUERIES_PER_BLOCK, MUSA_KV_TILE_ROWS,
        MUSA_SCORE_TILE_ROWS><<<blocks, GROUP_SIZE * QUERIES_PER_BLOCK>>>(
        d_q, d_k, d_v, d_o, params);
  } else {
    constexpr int GROUP_SIZE = 64;
    constexpr int QUERIES_PER_BLOCK = 8;
    const int target_tiles =
        musaPositiveCeilDiv(params.target_seq_len, QUERIES_PER_BLOCK);
    const int blocks = params.batch_size * params.query_heads * target_tiles;
    flashAttentionMusaFloatKVTiledKernel<
        64, GROUP_SIZE, QUERIES_PER_BLOCK, MUSA_FLOAT_HEAD64_KV_TILE_ROWS,
        MUSA_SCORE_TILE_ROWS><<<blocks, GROUP_SIZE * QUERIES_PER_BLOCK>>>(
        d_q, d_k, d_v, d_o, params);
  }
}

/**
 * @brief 根据 head_dim 启动 half MUSA K/V 分块 Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状和掩码参数
 */
inline void launchMusaKVTiled(const half* d_q, const half* d_k,
                              const half* d_v, half* d_o,
                              const MusaAttnParams& params) {
  // half 与 FP32 使用相同查询映射，但 D64 选择独立的 K/V tile 行数。
  if (params.head_dim == 32) {
    constexpr int GROUP_SIZE = 32;
    constexpr int QUERIES_PER_BLOCK = 16;
    const int target_tiles =
        musaPositiveCeilDiv(params.target_seq_len, QUERIES_PER_BLOCK);
    const int blocks = params.batch_size * params.query_heads * target_tiles;
    flashAttentionMusaHalfKVTiledKernel<
        half, 32, GROUP_SIZE, QUERIES_PER_BLOCK, MUSA_KV_TILE_ROWS,
        MUSA_SCORE_TILE_ROWS><<<blocks, GROUP_SIZE * QUERIES_PER_BLOCK>>>(
        d_q, d_k, d_v, d_o, params);
  } else {
    constexpr int GROUP_SIZE = 64;
    constexpr int QUERIES_PER_BLOCK = 8;
    const int target_tiles =
        musaPositiveCeilDiv(params.target_seq_len, QUERIES_PER_BLOCK);
    const int blocks = params.batch_size * params.query_heads * target_tiles;
    flashAttentionMusaHalfKVTiledKernel<
        half, 64, GROUP_SIZE, QUERIES_PER_BLOCK,
        MUSA_HALF_HEAD64_KV_TILE_ROWS,
        MUSA_SCORE_TILE_ROWS><<<blocks, GROUP_SIZE * QUERIES_PER_BLOCK>>>(
        d_q, d_k, d_v, d_o, params);
  }
}

/**
 * @brief 按头维度容量档位启动 FP32 MUSA 回退 Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状和掩码参数
 */
inline void launchMusaFallback(const float* d_q, const float* d_k,
                               const float* d_v, float* d_o,
                               const MusaAttnParams& params) {
  constexpr int THREADS = 128;
  const int total_queries =
      params.batch_size * params.target_seq_len * params.query_heads;
  const int blocks = musaPositiveCeilDiv(total_queries, THREADS);
  // 因果 D32 优先保持参考运算顺序，其余维度进入最小可容纳的累加器容量桶。
  if (params.is_causal && params.head_dim == 32) {
    flashAttentionMusaFloatReferenceOrderKernel<32><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else if (params.head_dim <= 32) {
    flashAttentionMusaFloatStrictKernel<32><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else if (params.head_dim <= 64) {
    flashAttentionMusaFloatStrictKernel<64><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else if (params.head_dim <= 128) {
    flashAttentionMusaFloatStrictKernel<128><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else if (params.head_dim <= 256) {
    flashAttentionMusaFloatStrictKernel<256><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else if (params.head_dim <= 512) {
    flashAttentionMusaFloatStrictKernel<512><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  } else {
    flashAttentionMusaFloatStrictKernel<1024><<<blocks, THREADS>>>(
        d_q, d_k, d_v, d_o, params);
  }
}

/**
 * @brief 按头维度容量档位启动 half MUSA 线程组回退 Kernel
 * @param d_q GPU 查询张量地址
 * @param d_k GPU 键张量地址
 * @param d_v GPU 值张量地址
 * @param d_o GPU 输出张量地址
 * @param params 注意力形状和掩码参数
 */
inline void launchMusaFallback(const half* d_q, const half* d_k,
                               const half* d_v, half* d_o,
                               const MusaAttnParams& params) {
  constexpr int GROUP_SIZE = MUSA_GROUP_SIZE;
  constexpr int GROUPS_PER_BLOCK = MUSA_GROUPS_PER_BLOCK;
  constexpr int TILE = MUSA_K_TILE;
  constexpr int MAX_HEAD_DIM = 1024;
  // 网格布局为 [Q 头组, 目标位置, batch]，block 内所有组拥有统一因果边界。
  const dim3 block(GROUP_SIZE * GROUPS_PER_BLOCK);
  const dim3 grid(
      static_cast<unsigned int>(
          musaPositiveCeilDiv(params.query_heads, GROUPS_PER_BLOCK)),
      static_cast<unsigned int>(params.target_seq_len),
      static_cast<unsigned int>(params.batch_size));
  // 选择不小于实际 head_dim 的模板容量，控制每个 lane 的寄存器数组大小。
  if (params.head_dim <= GROUP_SIZE) {
    flashAttentionGroupKernel<half, GROUP_SIZE, GROUPS_PER_BLOCK, GROUP_SIZE,
                              TILE><<<grid, block>>>(
        d_q, d_k, d_v, d_o, params.target_seq_len, params.src_seq_len,
        params.query_heads, params.kv_heads, params.head_dim, params.is_causal);
  } else if (params.head_dim <= 2 * GROUP_SIZE) {
    flashAttentionGroupKernel<half, GROUP_SIZE, GROUPS_PER_BLOCK,
                              2 * GROUP_SIZE, TILE><<<grid, block>>>(
        d_q, d_k, d_v, d_o, params.target_seq_len, params.src_seq_len,
        params.query_heads, params.kv_heads, params.head_dim, params.is_causal);
  } else if (params.head_dim <= 4 * GROUP_SIZE) {
    flashAttentionGroupKernel<half, GROUP_SIZE, GROUPS_PER_BLOCK,
                              4 * GROUP_SIZE, TILE><<<grid, block>>>(
        d_q, d_k, d_v, d_o, params.target_seq_len, params.src_seq_len,
        params.query_heads, params.kv_heads, params.head_dim, params.is_causal);
  } else {
    flashAttentionGroupKernel<half, GROUP_SIZE, GROUPS_PER_BLOCK, MAX_HEAD_DIM,
                              TILE><<<grid, block>>>(
        d_q, d_k, d_v, d_o, params.target_seq_len, params.src_seq_len,
        params.query_heads, params.kv_heads, params.head_dim, params.is_causal);
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
  // Host 输入按 [rows, hidden_dim] 展平，乘法通过辅助函数检查 size_t 溢出。
  const size_t input_elems =
      checkedMusaElementProduct(rows, hidden_dim, "rmsNorm input");
  if (h_input.size() != input_elems || h_weight.size() != hidden_dim) {
    throw std::invalid_argument("rmsNorm: input or weight shape mismatch");
  }
  h_output.resize(input_elems);
  if (input_elems == 0) {
    return;
  }

  // 一个 block 对应一行，因此 rows 必须落在设备 grid.x 上限内。
  const MusaRuntimeDeviceInfo& device = currentMusaDeviceInfo();
  if (rows > static_cast<size_t>(device.max_grid_x)) {
    throw std::overflow_error("rmsNorm: row count exceeds MUSA grid limit");
  }

  // 重复调用时复用输入、权重和输出显存，并保证缓冲区属于当前设备。
  static thread_local ReusableDeviceBuffer<T> input_buffer;
  static thread_local ReusableDeviceBuffer<T> weight_buffer;
  static thread_local ReusableDeviceBuffer<T> output_buffer;

  input_buffer.ensure(input_elems, device.device_id);
  weight_buffer.ensure(hidden_dim, device.device_id);
  output_buffer.ensure(input_elems, device.device_id);

  T* d_input = input_buffer.data();
  T* d_weight = weight_buffer.data();
  T* d_output = output_buffer.data();

  // 将输入和逐列权重传到设备；Kernel 结果直接写入 d_output。
  RUNTIME_CHECK(musaMemcpy(d_input, h_input.data(), input_elems * sizeof(T),
                         musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),
                         musaMemcpyHostToDevice));

  // 选择不超过 256、设备线程上限和 hidden_dim 的最大 2 的幂，满足树形归约要求。
  int threads = 1;
  while (threads * 2 <= std::min(256, device.max_threads_per_block) &&
         static_cast<size_t>(threads * 2) <= hidden_dim) {
    threads *= 2;
  }
  // 动态共享内存大小等于线程数乘一个 FP32 局部平方和槽位。
  const dim3 blocks(static_cast<unsigned int>(rows));
  const size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);
  rmsNormKernel<T><<<blocks, threads, shared_mem>>>(d_input, d_weight, d_output,
                                                    rows, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());

  // 默认流中的阻塞 D2H 拷贝会等待 Kernel 完成，无需额外设备同步。
  RUNTIME_CHECK(musaMemcpy(h_output.data(), d_output, input_elems * sizeof(T),
                         musaMemcpyDeviceToHost));
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
  // 检查基本形状和 GQA 约束：Q 头数必须能被 K/V 头数整除。
  if (batch_size < 0 || target_seq_len < 0 || src_seq_len < 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument(
        "flashAttention: invalid non-positive shape parameter");
  }
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument(
        "flashAttention: query_heads must be divisible by kv_heads");
  }

  constexpr int MAX_HEAD_DIM = 1024;

  // 当前所有回退 Kernel 的最大模板容量为 1024。
  if (head_dim > MAX_HEAD_DIM) {
    throw std::invalid_argument(
        "flashAttention: head_dim exceeds maximum supported value");
  }

  // 计算 Q/K/V 元素数量并检查溢出；因果模式截断设备端不可能访问的 K/V 后缀。
  const size_t q_elems = checkedMusaElementProduct(
      checkedMusaElementProduct(
          checkedMusaElementProduct(static_cast<size_t>(batch_size),
                                    static_cast<size_t>(target_seq_len),
                                    "query"),
          static_cast<size_t>(query_heads), "query"),
      static_cast<size_t>(head_dim), "query");
  const size_t kv_elems = checkedMusaElementProduct(
      checkedMusaElementProduct(
          checkedMusaElementProduct(static_cast<size_t>(batch_size),
                                    static_cast<size_t>(src_seq_len),
                                    "key/value"),
          static_cast<size_t>(kv_heads), "key/value"),
      static_cast<size_t>(head_dim), "key/value");
  const int device_src_seq_len =
      is_causal ? ((target_seq_len < src_seq_len) ? target_seq_len : src_seq_len)
                : src_seq_len;
  const size_t device_kv_elems = checkedMusaElementProduct(
      checkedMusaElementProduct(
          checkedMusaElementProduct(static_cast<size_t>(batch_size),
                                    static_cast<size_t>(device_src_seq_len),
                                    "device key/value"),
          static_cast<size_t>(kv_heads), "device key/value"),
      static_cast<size_t>(head_dim), "device key/value");
  const size_t total_queries = q_elems / static_cast<size_t>(head_dim);
  if (total_queries > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("flashAttention: query count exceeds int range");
  }
  if (h_q.size() != q_elems || h_k.size() != kv_elems ||
      h_v.size() != kv_elems) {
    throw std::invalid_argument("flashAttention: input tensor shape mismatch");
  }
  // Host 输出最终会被 D2H 覆盖，因此禁止与任一输入 vector 是同一对象。
  if (&h_o == &h_q || &h_o == &h_k || &h_o == &h_v) {
    throw std::invalid_argument("flashAttention: output must not alias input");
  }
  // 输出布局与 Q 完全相同；空输入直接返回对应大小的零向量。
  h_o.resize(q_elems);
  if (q_elems == 0 || kv_elems == 0) {
    std::fill(h_o.begin(), h_o.end(), fromFloat<T>(0.0f));
    return;
  }

  // 查询设备资源并检查三维网格范围，后续路径选择也依赖线程和共享内存上限。
  const MusaRuntimeDeviceInfo& device = currentMusaDeviceInfo();
  if (target_seq_len > device.max_grid_y || batch_size > device.max_grid_z) {
    throw std::overflow_error(
        "flashAttention: shape exceeds MUSA multidimensional grid limit");
  }

  // 每个 Host 线程复用独立的 Q/K/V/O 设备缓冲区。
  static thread_local ReusableDeviceBuffer<T> q_buffer;
  static thread_local ReusableDeviceBuffer<T> k_buffer;
  static thread_local ReusableDeviceBuffer<T> v_buffer;
  static thread_local ReusableDeviceBuffer<T> o_buffer;

  // K/V 缓冲区按截断后的 device_src_seq_len 分配，可小于 Host 原始张量。
  q_buffer.ensure(q_elems, device.device_id);
  k_buffer.ensure(device_kv_elems, device.device_id);
  v_buffer.ensure(device_kv_elems, device.device_id);
  o_buffer.ensure(q_elems, device.device_id);

  T* d_q = q_buffer.data();
  T* d_k = k_buffer.data();
  T* d_v = v_buffer.data();
  T* d_o = o_buffer.data();

  // 复制 Q；因果模式通过二维拷贝逐 batch 压缩 K/V 前缀。
  RUNTIME_CHECK(
      musaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), musaMemcpyHostToDevice));
  if (device_src_seq_len == src_seq_len) {
    RUNTIME_CHECK(
        musaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T), musaMemcpyHostToDevice));
    RUNTIME_CHECK(
        musaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T), musaMemcpyHostToDevice));
  } else {
    const size_t host_batch_pitch =
        static_cast<size_t>(src_seq_len) * kv_heads * head_dim * sizeof(T);
    const size_t device_batch_pitch = static_cast<size_t>(device_src_seq_len) *
                                      kv_heads * head_dim * sizeof(T);
    RUNTIME_CHECK(musaMemcpy2D(d_k, device_batch_pitch, h_k.data(),
                               host_batch_pitch, device_batch_pitch,
                               batch_size, musaMemcpyHostToDevice));
    RUNTIME_CHECK(musaMemcpy2D(d_v, device_batch_pitch, h_v.data(),
                               host_batch_pitch, device_batch_pitch,
                               batch_size, musaMemcpyHostToDevice));
  }

  // 先根据数据类型和规模判断分块收益，再结合设备线程数和共享内存判断能否启动。
  const int tiled_min_sequence =
      std::is_same<T, float>::value ? MUSA_FLOAT_KV_TILED_MIN_SEQUENCE
                                    : MUSA_HALF_KV_TILED_MIN_SEQUENCE;
  // 不同数据类型和头维度对应不同静态共享内存占用，必须小于设备上限。
  size_t kv_tiled_shared_mem = 0;
  if (head_dim == 32) {
    kv_tiled_shared_mem =
        std::is_same<T, float>::value ? 18944u : 25600u;
  } else if (head_dim == 64) {
    kv_tiled_shared_mem =
        std::is_same<T, float>::value ? 18688u : 25088u;
  }
  // 通用 K/V 分块要求非因果 FP32或 half、大规模输入、512 线程能力和足够共享内存。
  const bool use_kv_tiled =
      !(std::is_same<T, float>::value && is_causal) &&
      shouldUseMusaKVTiled(total_queries, target_seq_len, device_src_seq_len,
                           head_dim, tiled_min_sequence) &&
      device.max_threads_per_block >= 512 &&
      kv_tiled_shared_mem <= device.shared_mem_per_block;
  // 打包原始形状和 GQA 派生参数，按值传给所有设备 Kernel。
  const MusaAttnParams params{batch_size,
                              target_seq_len,
                              device_src_seq_len,
                              query_heads,
                              kv_heads,
                              head_dim,
                              query_heads / kv_heads,
                              is_causal};

  // 路径优先级：因果 FP32 D32 查询分块、通用 K/V 分块、类型专用回退。
  const bool used_float_d32_query_tile =
      launchMusaFloatD32QueryTiledIfEligible(
          d_q, d_k, d_v, d_o, params, total_queries, device);
  if (used_float_d32_query_tile) {
    // 类型专用重载已经完成 Kernel 启动，此分支无需重复操作。
  } else if (use_kv_tiled) {
    launchMusaKVTiled(d_q, d_k, d_v, d_o, params);
  } else {
    launchMusaFallback(d_q, d_k, d_v, d_o, params);
  }

  RUNTIME_CHECK(musaGetLastError());

  // 默认流中的阻塞 D2H 拷贝会等待 Kernel 完成。
  RUNTIME_CHECK(
      musaMemcpy(h_o.data(), d_o, q_elems * sizeof(T), musaMemcpyDeviceToHost));
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
