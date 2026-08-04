#pragma once

#include <infiniband/verbs.h>

#include <cstdint>

#include "micro_ccl/verbs/completion_queue.hpp"
#include "micro_ccl/verbs/device.hpp"
#include "micro_ccl/verbs/endpoint_info.hpp"
#include "micro_ccl/verbs/protection_domain.hpp"

namespace micro_ccl {

// An RC (Reliable Connection) queue pair: a send queue and a receive queue
// bound to one specific remote QP, with hardware-level reliability --
// in-order delivery, ACKs, and retransmission are handled by the NIC/driver,
// the same guarantees TCP gives you but implemented in the RDMA transport
// instead of the kernel network stack. This is why micro-ccl can treat
// point-to-point send/recv as "just works" rather than building its own
// ARQ protocol: RC already did that job. (The other verbs transport modes
// -- UD, unreliable/datagram -- trade that guarantee for lower per-message
// overhead; RC is the right choice here because collectives need every
// message to arrive, in order, or the algorithm's correctness breaks.)
//
// A freshly created QP starts in state RESET and is useless until walked
// through RESET -> INIT -> RTR -> RTS. INIT is local-only setup (which
// port, what access to allow). RTR ("ready to receive") is where the QP
// learns about its peer -- remote QP number, remote address (LID or GID),
// negotiated MTU -- which is exactly the information the bootstrap/
// out-of-band TCP exchange collects. RTS ("ready to send") adds this side's
// own starting sequence number and retry/timeout policy. Only after RTS can
// either side post a send.
class QueuePair {
 public:
  QueuePair(ProtectionDomain& pd, CompletionQueue& send_cq,
            CompletionQueue& recv_cq, uint32_t max_send_wr = 64,
            uint32_t max_recv_wr = 64, uint32_t max_sge = 1);
  ~QueuePair();

  QueuePair(const QueuePair&) = delete;
  QueuePair& operator=(const QueuePair&) = delete;
  QueuePair(QueuePair&& other) noexcept;
  QueuePair& operator=(QueuePair&& other) noexcept;

  uint32_t qp_num() const { return qp_->qp_num; }
  ibv_qp* native() const { return qp_; }

  // A random 24-bit starting PSN (packet sequence numbers are a 24-bit
  // field in the RDMA spec). Randomizing rather than always starting at 0
  // is standard practice (matches ib_send_bw/rc_pingpong) -- it means a
  // stale retransmitted packet from a previous run of this same QP number
  // is far less likely to be mistaken for a valid packet in a new
  // connection.
  static uint32_t generate_psn();

  // RESET -> INIT: local-only. Declares which port this QP will use and
  // what remote access to allow once connected.
  void modify_to_init(const Device& device);

  // INIT -> RTR: supplies everything learned about the peer via bootstrap.
  // local_device is needed here (not just at construction) because the
  // address-handle path differs by device: InfiniBand addresses the peer
  // by LID, RoCE by GID, and only Device knows which this is.
  void modify_to_rtr(const Device& local_device,
                      const QpEndpointInfo& remote);

  // RTR -> RTS: local_psn must be the same value advertised to the peer in
  // this side's QpEndpointInfo (the peer's modify_to_rtr used it as their
  // expected receive sequence number).
  void modify_to_rts(uint32_t local_psn);

  // Snapshot of this QP's connection info to send to the peer over
  // bootstrap. Must be called after construction (qp_num is known
  // immediately) but the psn passed in should be whatever this side intends
  // to use as its RTS sq_psn.
  QpEndpointInfo local_info(const Device& device, uint32_t rank,
                             uint32_t local_psn) const;

 private:
  ibv_qp* qp_ = nullptr;
};

}  // namespace micro_ccl
