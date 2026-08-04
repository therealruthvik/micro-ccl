#include "micro_ccl/transport.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "micro_ccl/detail/error.hpp"

namespace micro_ccl::transport {

void post_send(QueuePair& qp, const MemoryRegion& mr, const void* data,
                size_t len, uint64_t wr_id, bool signaled) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(data);
  sge.length = static_cast<uint32_t>(len);
  sge.lkey = mr.lkey();

  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;

  ibv_send_wr* bad_wr = nullptr;
  // ibv_post_send returns the errno value directly (0 on success), the same
  // convention as ibv_modify_qp -- it does not go through the global errno.
  check_rc(ibv_post_send(qp.native(), &wr, &bad_wr), "ibv_post_send");
}

void post_recv(QueuePair& qp, const MemoryRegion& mr, void* data, size_t len,
                uint64_t wr_id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(data);
  sge.length = static_cast<uint32_t>(len);
  sge.lkey = mr.lkey();

  ibv_recv_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_recv_wr* bad_wr = nullptr;
  check_rc(ibv_post_recv(qp.native(), &wr, &bad_wr), "ibv_post_recv");
}

ibv_wc wait_completion(CompletionQueue& cq) {
  std::vector<ibv_wc> wcs;
  // Busy-polling rather than the event-channel API (ibv_req_notify_cq +
  // blocking on a file descriptor) trades CPU for latency: at the message
  // sizes and rates this library targets, the wakeup latency of the
  // event-driven path would dominate the measurement we're trying to take.
  // Production NIC drivers/CCLs make the same tradeoff for latency-
  // sensitive small messages.
  for (;;) {
    int n = cq.poll(wcs, 1);
    if (n > 0) {
      break;
    }
  }
  const ibv_wc& wc = wcs.front();
  if (wc.status != IBV_WC_SUCCESS) {
    throw std::runtime_error(std::string("completion error: ") +
                              ibv_wc_status_str(wc.status));
  }
  return wc;
}

void send(QueuePair& qp, CompletionQueue& send_cq, const MemoryRegion& mr,
          const void* data, size_t len) {
  post_send(qp, mr, data, len, /*wr_id=*/0, /*signaled=*/true);
  wait_completion(send_cq);
}

size_t recv(QueuePair& qp, CompletionQueue& recv_cq, const MemoryRegion& mr,
            void* data, size_t capacity) {
  post_recv(qp, mr, data, capacity, /*wr_id=*/0);
  ibv_wc wc = wait_completion(recv_cq);
  return wc.byte_len;
}

}  // namespace micro_ccl::transport
