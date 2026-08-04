#include "micro_ccl/collectives/reduce_ops.hpp"

#include <algorithm>
#include <stdexcept>

namespace micro_ccl {
namespace {

// Templated on element type T so the same three-line loop body works for
// both float and int32_t without duplicating it -- the compiler generates
// one specialized version per T, so there is no runtime dispatch cost
// inside the loop, only the one outer switch in reduce_inplace() below.
template <typename T>
void reduce_inplace_typed(T* dst, const T* src, size_t count, ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum:
      for (size_t i = 0; i < count; ++i) dst[i] = dst[i] + src[i];
      return;
    case ReduceOp::Min:
      for (size_t i = 0; i < count; ++i) dst[i] = std::min(dst[i], src[i]);
      return;
    case ReduceOp::Max:
      for (size_t i = 0; i < count; ++i) dst[i] = std::max(dst[i], src[i]);
      return;
  }
  throw std::invalid_argument("reduce_inplace: unknown ReduceOp");
}

}  // namespace

void reduce_inplace(void* dst, const void* src, size_t count, Dtype dtype,
                     ReduceOp op) {
  switch (dtype) {
    case Dtype::Float32:
      reduce_inplace_typed(static_cast<float*>(dst),
                            static_cast<const float*>(src), count, op);
      return;
    case Dtype::Int32:
      reduce_inplace_typed(static_cast<int32_t*>(dst),
                            static_cast<const int32_t*>(src), count, op);
      return;
  }
  throw std::invalid_argument("reduce_inplace: unknown Dtype");
}

}  // namespace micro_ccl
