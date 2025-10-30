# Warp Scheduling

## 🧩 Note: Warp Execution, Scheduling, and SM Microarchitecture

### Streaming Multiprocessor (SM) Overview

* An **SM** is the fundamental execution unit of an NVIDIA GPU.
* It hosts:

  * **Warp schedulers** (typically 4 per SM in recent architectures).
  * **Dispatch units**, **execution pipelines** (INT, FP32, FP64, SFU, tensor units, etc.).
  * **Register file**, **shared memory/L1 cache**, and control logic.
* Each SM can keep **dozens of warps** (32–64 typically) *resident* at once — depending on resource limits (registers, shared memory, block size).

Each **warp = 32 threads** executing in SIMT (Single Instruction, Multiple Thread) style.

---

### Warp Scheduling Model

* A **warp scheduler** is responsible for managing a subset of active warps within the SM.
* Every cycle, each scheduler:

  1. **Selects one ready warp** from its pool (those not waiting on memory or dependencies).
  2. Issues **one instruction** from that warp to its **dispatch unit**.
  3. The dispatch unit routes it to the appropriate **execution pipeline** (e.g., FP32, INT32, SFU, tensor core).

> ⏱️ The GPU uses **fine-grained multithreading** — each warp is logically independent but shares hardware pipelines temporally.

---

### Instruction Interleaving and Concurrency

* The SM maintains **many active warps** concurrently.
* Each warp has its own **program counter (PC)**, so instruction streams are independent.
* Even if no stalls occur (pure compute kernel, no memory latency):

  * The scheduler **still interleaves** instructions from different warps rather than running one warp to completion.
  * This interleaving avoids pipeline bubbles and maximizes throughput.
  * The scheduling policy may be **round-robin**, **GTO (Greedy-Then-Oldest)**, or other latency-hiding heuristics depending on architecture.

In other words, the scheduler time-slices between warps at the **instruction granularity**, not at kernel completion.

---

### Why the Warp Doesn’t Run to Completion

Even with no memory stalls, there are still:

* **Pipeline latencies** (e.g., an FP32 multiply has a latency of several cycles).
* **Functional unit constraints** (limited number of ALUs or SFUs per cycle).
* **Dependency chains** (some instructions must wait on prior results).

To cover these, the scheduler issues instructions from *other* ready warps while a warp’s earlier instruction is still in flight.

Thus, **no single warp monopolizes** the scheduler; all active warps are interleaved dynamically.

---

### The Role of the Fetch/Dispatch Unit

* The **fetch/decode/dispatch logic** manages the instruction flow for *all* warps in the SM.
* It doesn’t “replay” the same instruction stream serially.
  Instead:

  * Each warp’s PC determines which instruction to fetch.
  * The control logic maintains a small set of per-warp PCs and instruction buffers.
  * Each scheduler fetches the instruction corresponding to the warp it selects that cycle.

This creates a **concurrent control flow** inside the SM — many instruction streams coexisting, but only a few being issued each cycle.

---

### Warp Completion and Context Lifetime

* A **warp remains active** on the SM until:

  1. All its threads have completed execution, and
  2. The **thread block** it belongs to has finished.
* Only then are its hardware resources (register file space, shared memory) released, freeing slots for new warps or thread blocks to be launched.

This ensures zero-cost context switching — all warps in an SM remain *resident* and *ready* to execute at any cycle.

---

### Conceptual Timeline (Example)

Assume two warps, `W1` and `W2`, sharing one scheduler:

| Cycle | Scheduler Action         | Comment                      |
| ----- | ------------------------ | ---------------------------- |
| 1     | Issue `FADD` from W1     | Ready warp chosen            |
| 2     | Issue `FMUL` from W2     | Interleaved to cover latency |
| 3     | Issue next `FMA` from W1 | W1 result now available      |
| 4     | Issue `ADD` from W2      | Keeps both warps progressing |
| 5     | Issue `FMA` from W1      | Continues pipeline overlap   |

Even though neither warp stalls on memory, their instructions are still interleaved across cycles.

If `W1` later hits a memory load, `W2` continues without pause.

---

### Summary Table

| Concept            | Behavior                                          |
| ------------------ | ------------------------------------------------- |
| Warp size          | 32 threads (SIMT unit)                            |
| Scheduler capacity | Multiple warps (e.g., 8–16 per scheduler)         |
| Issue rate         | 1 instruction per warp per cycle per scheduler    |
| Execution model    | Fine-grained interleaving (not run-to-completion) |
| Warp PC tracking   | Independent per warp                              |
| Latency hiding     | By switching among ready warps                    |
| Warp lifetime      | Until block completion                            |
| Scheduling policy  | Round-robin / GTO / architecture-dependent        |

---

### 🧠 Key Insight

> The GPU’s strength lies in **latency hiding via massive concurrency**, not in fast single-thread execution.
> An SM’s warp schedulers behave like *microthreaded pipelines* — constantly feeding execution units from a pool of warps, overlapping computation, memory access, and instruction latency at instruction-level granularity.

---
