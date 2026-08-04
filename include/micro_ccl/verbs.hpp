#pragma once

// Umbrella header for the RAII verbs layer. Application code (transport,
// collectives, benchmarks) includes this rather than reaching for
// individual ibv_* headers or handles directly -- raw ibv_* types stop at
// this layer.
#include "micro_ccl/verbs/completion_queue.hpp"
#include "micro_ccl/verbs/device.hpp"
#include "micro_ccl/verbs/endpoint_info.hpp"
#include "micro_ccl/verbs/memory_region.hpp"
#include "micro_ccl/verbs/protection_domain.hpp"
#include "micro_ccl/verbs/queue_pair.hpp"
