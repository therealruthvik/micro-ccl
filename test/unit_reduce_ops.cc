// Unit tests for the reduction operators. These are pure functions over
// plain memory -- no verbs, no network -- so unlike the integration tests
// in this directory, they run anywhere: no RDMA hardware, no Soft-RoCE,
// plain `ctest` on any machine including CI.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "micro_ccl/collectives/reduce_ops.hpp"

using namespace micro_ccl;

TEST_CASE("dtype_size reports the element width used on the wire") {
  REQUIRE(dtype_size(Dtype::Float32) == 4);
  REQUIRE(dtype_size(Dtype::Int32) == 4);
}

TEST_CASE("reduce_inplace sums float32 element-wise") {
  std::vector<float> dst = {1.0f, 2.0f, -3.5f};
  std::vector<float> src = {10.0f, -5.0f, 0.5f};
  reduce_inplace(dst.data(), src.data(), dst.size(), Dtype::Float32,
                  ReduceOp::Sum);
  REQUIRE(dst[0] == 11.0f);
  REQUIRE(dst[1] == -3.0f);
  REQUIRE(dst[2] == -3.0f);
}

TEST_CASE("reduce_inplace takes min/max of float32") {
  std::vector<float> dst = {5.0f, -1.0f, 3.0f};
  std::vector<float> src = {2.0f, -4.0f, 3.0f};

  std::vector<float> min_result = dst;
  reduce_inplace(min_result.data(), src.data(), dst.size(), Dtype::Float32,
                  ReduceOp::Min);
  REQUIRE(min_result[0] == 2.0f);
  REQUIRE(min_result[1] == -4.0f);
  REQUIRE(min_result[2] == 3.0f);

  std::vector<float> max_result = dst;
  reduce_inplace(max_result.data(), src.data(), dst.size(), Dtype::Float32,
                  ReduceOp::Max);
  REQUIRE(max_result[0] == 5.0f);
  REQUIRE(max_result[1] == -1.0f);
  REQUIRE(max_result[2] == 3.0f);
}

TEST_CASE("reduce_inplace sums int32 including negative values") {
  std::vector<int32_t> dst = {100, -50, 0};
  std::vector<int32_t> src = {-30, -50, 42};
  reduce_inplace(dst.data(), src.data(), dst.size(), Dtype::Int32,
                  ReduceOp::Sum);
  REQUIRE(dst[0] == 70);
  REQUIRE(dst[1] == -100);
  REQUIRE(dst[2] == 42);
}

TEST_CASE("reduce_inplace min/max int32") {
  std::vector<int32_t> dst = {7, -2, 9};
  std::vector<int32_t> src = {3, -8, 20};

  std::vector<int32_t> min_result = dst;
  reduce_inplace(min_result.data(), src.data(), dst.size(), Dtype::Int32,
                  ReduceOp::Min);
  REQUIRE(min_result[0] == 3);
  REQUIRE(min_result[1] == -8);
  REQUIRE(min_result[2] == 9);

  std::vector<int32_t> max_result = dst;
  reduce_inplace(max_result.data(), src.data(), dst.size(), Dtype::Int32,
                  ReduceOp::Max);
  REQUIRE(max_result[0] == 7);
  REQUIRE(max_result[1] == -2);
  REQUIRE(max_result[2] == 20);
}

TEST_CASE("reduce_inplace with count=0 touches nothing") {
  std::vector<int32_t> dst = {42};
  std::vector<int32_t> src = {999};
  reduce_inplace(dst.data(), src.data(), 0, Dtype::Int32, ReduceOp::Sum);
  REQUIRE(dst[0] == 42);  // unchanged -- count=0 must be a true no-op
}
