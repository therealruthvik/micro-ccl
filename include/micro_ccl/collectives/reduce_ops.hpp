#pragma once

#include <cstddef>
#include <cstdint>

namespace micro_ccl {

// The two element types the collectives operate on. Kept as a closed set
// (not a template parameter threaded through every public API) so
// Communicator/collective call sites can be ordinary functions taking a
// runtime enum -- the alternative, templating broadcast/allreduce
// themselves on element type, would force every call site to know its
// dtype at compile time, which the benchmark harness's CLI-driven sweep
// does not.
enum class Dtype { Float32, Int32 };

enum class ReduceOp { Sum, Min, Max };

inline size_t dtype_size(Dtype dtype) {
  switch (dtype) {
    case Dtype::Float32:
      return sizeof(float);
    case Dtype::Int32:
      return sizeof(int32_t);
  }
  return 0;  // unreachable; silences -Wreturn-type without a default case
             // that would suppress -Wswitch catching a future enum value.
}

// dst[i] = op(dst[i], src[i]) for i in [0, count). dst and src are raw
// bytes because that's what arrives off the wire (a received chunk in a
// pre-registered buffer); the dtype tag is what lets this function safely
// reinterpret them as float/int32_t internally.
void reduce_inplace(void* dst, const void* src, size_t count, Dtype dtype,
                     ReduceOp op);

}  // namespace micro_ccl
