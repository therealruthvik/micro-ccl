#pragma once

#include <cstddef>
#include <cstdint>

#include "micro_ccl/verbs.hpp"

namespace micro_ccl::transport {

// Two-sided send/recv over an already-RTS QueuePair. "Two-sided" means both
// ends actively participate -- the sender posts a SEND, the receiver must
// have already posted a matching RECV, exactly like MPI_Send/MPI_Recv or a
// plain TCP socket. This is deliberately not the one-sided RDMA read/write
// path (where the initiator touches remote memory without the peer's CPU
// doing anything) -- collectives here use two-sided semantics because they
// need a *notification* that data arrived, not just the data itself, and
// building that notification on top of one-sided RDMA (immediate data,
// polling a flag) is strictly more complexity for no benefit at this
// message count.
//
// Every call here takes a MemoryRegion because the local key (lkey) it
// carries is what allows the NIC to touch that memory directly -- there is
// no copy into a staging buffer anywhere in this header, by construction.

// Posts a SEND work request for [data, data+len) — the caller is
// responsible for len bytes at contains within mr's registered range.
// signaled=true (the default) means a completion will show up on qp's send
// CQ once the NIC confirms the peer's receive queue accepted the message;
// callers that intentionally do not want to wait per-message (see the ring
// allreduce, which pipelines sends) can pass false and drain completions in
// bulk instead.
void post_send(QueuePair& qp, const MemoryRegion& mr, const void* data,
                size_t len, uint64_t wr_id, bool signaled = true);

// Posts a receive buffer to catch the next SEND arriving on qp. Capacity is
// `len` bytes at `data` -- if the peer's SEND is larger than this, the RC
// transport itself refuses to overrun the buffer: the NIC raises a length-
// error completion (caught and turned into an exception by
// wait_completion()) instead of writing past the posted region. That is a
// hardware-level guarantee, not something this wrapper has to implement.
void post_recv(QueuePair& qp, const MemoryRegion& mr, void* data, size_t len,
                uint64_t wr_id);

// Busy-polls cq until at least one completion is available and returns it.
// Throws if the completion's status is not IBV_WC_SUCCESS -- a failed send
// or recv is surfaced as an exception here, not as a silently-wrong wc the
// caller has to remember to check.
ibv_wc wait_completion(CompletionQueue& cq);

// Convenience synchronous wrappers (post + wait in one call) for code paths
// where overlap doesn't matter -- the pingpong example and simple
// broadcast use these. Collectives with pipelining (ring allreduce) call
// post_send/post_recv directly instead.
void send(QueuePair& qp, CompletionQueue& send_cq, const MemoryRegion& mr,
          const void* data, size_t len);

// Returns the number of bytes actually received (wc.byte_len), which can be
// less than `capacity` but never more -- see post_recv's note on length
// errors. Callers that require an exact size (e.g. collectives, where a
// short receive means a peer disagreed about chunk size) must check the
// return value themselves; this function only guarantees memory safety, not
// application-level size agreement.
size_t recv(QueuePair& qp, CompletionQueue& recv_cq, const MemoryRegion& mr,
            void* data, size_t capacity);

}  // namespace micro_ccl::transport
