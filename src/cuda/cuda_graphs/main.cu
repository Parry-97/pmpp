#include <iostream>
#include <stdio.h>

#define N 500000

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err),           \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief CUDA kernel for cuda_graphs
 */
__global__ void short_kernel(float *input, float *output, int chunk_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < chunk_size) {
    output[i] = input[i] + 1.23f;
  }
}

/**
 * @brief Host wrapper function for cuda_graphs
 */
void cuda_graphs_pipelined(float *h_input, float *h_output) {
  const int num_streams = 4;
  const int chunk_size = N / num_streams;
  const int chunk_bytes = chunk_size * sizeof(float);

  float *d_input, *d_output;
  cudaStream_t streams[num_streams];

  // Allocate device memory normally
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(float)));

  // Create our army of streams
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }

  // --- THE PIPELINE LOOP ---
  for (int i = 0; i < num_streams; ++i) {
    int offset = i * chunk_size;

    // 1. Asynchronously copy this chunk to the device
    CUDA_CHECK(cudaMemcpyAsync(d_input + offset, h_input + offset, chunk_bytes,
                               cudaMemcpyHostToDevice, streams[i]));

    // 2. Launch kernel on this chunk immediately after its copy schedules
    int threadsPerBlock = 256;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    short_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
        d_input + offset, d_output + offset,
        chunk_size // Note: Modify kernel to take chunk_size
    );

    // 3. Asynchronously copy this chunk back to the host
    CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output + offset,
                               chunk_bytes, cudaMemcpyDeviceToHost,
                               streams[i]));
  }

  // Synchronize all streams before cleaning up
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void cuda_graphs_captured(float *h_input, float *h_output) {
  const int num_streams = 4;
  const int chunk_size = N / num_streams;
  const int chunk_bytes = chunk_size * sizeof(float);

  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(float)));

  // Create a main stream for managing capture, and an array of worker streams
  cudaStream_t main_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));

  cudaStream_t streams[num_streams];
  cudaEvent_t join_events[num_streams];
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    CUDA_CHECK(
        cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
  }

  cudaEvent_t fork_event;
  CUDA_CHECK(cudaEventCreate(&fork_event));

  // --- 1. START STREAM CAPTURE ---
  // Using Global mode allows the capture to cross over into our worker streams
  CUDA_CHECK(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal));

  // FORK: Record an event on main_stream and make all workers wait on it
  CUDA_CHECK(cudaEventRecord(fork_event, main_stream));

  for (int i = 0; i < num_streams; ++i) {
    int offset = i * chunk_size;

    // Make the worker stream wait until the fork point is captured
    CUDA_CHECK(cudaStreamWaitEvent(streams[i], fork_event, 0));

    // Queue pipelined operations into the worker stream
    CUDA_CHECK(cudaMemcpyAsync(d_input + offset, h_input + offset, chunk_bytes,
                               cudaMemcpyHostToDevice, streams[i]));

    int threadsPerBlock = 256;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    short_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
        d_input + offset, d_output + offset, chunk_size);

    CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output + offset,
                               chunk_bytes, cudaMemcpyDeviceToHost,
                               streams[i]));

    // Record a join event for this worker stream
    CUDA_CHECK(cudaEventRecord(join_events[i], streams[i]));
  }

  // JOIN: Make the main stream wait for all worker stream operations to
  // complete
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, join_events[i], 0));
  }

  // --- 2. END CAPTURE & INSTANTIATE ---
  cudaGraph_t graph;
  cudaGraphExec_t graph_instance;
  CUDA_CHECK(cudaStreamEndCapture(main_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));

  // --- 3. EXECUTE THE GRAPH ---
  // This single call triggers the entire parallel, multi-stream pipeline!
  CUDA_CHECK(cudaGraphLaunch(graph_instance, main_stream));
  CUDA_CHECK(cudaGraphLaunch(graph_instance, main_stream));
  CUDA_CHECK(cudaStreamSynchronize(main_stream));

  // --- Clean up ---
  CUDA_CHECK(cudaGraphExecDestroy(graph_instance));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(fork_event));
  CUDA_CHECK(cudaStreamDestroy(main_stream));
  for (int i = 0; i < num_streams; ++i) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaEventDestroy(join_events[i]));
  }
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void print_array(const std::string &label, float *v, int n, int limit = 5) {
  std::cout << label << ": ";
  for (int i = 0; i < n && i < limit; i++) {
    std::cout << v[i] << " ";
  }
  if (n > limit)
    std::cout << "... (and " << (n - limit) << " more)\n";
}

void queryDevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for (int i = 0; i < numDevices; i++) {
    cudaSetDevice(i);
    cudaInitDevice(0, 0, 0);
    int deviceId = i;
    int concurrentManagedAccess = -1;
    cudaDeviceGetAttribute(&concurrentManagedAccess,
                           cudaDevAttrConcurrentManagedAccess, deviceId);
    int pageableMemoryAccess = -1;
    cudaDeviceGetAttribute(&pageableMemoryAccess,
                           cudaDevAttrPageableMemoryAccess, deviceId);
    int pageableMemoryAccessUsesHostPageTables = -1;
    cudaDeviceGetAttribute(&pageableMemoryAccessUsesHostPageTables,
                           cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                           deviceId);
    printf("Device %d has ", deviceId);
    if (concurrentManagedAccess) {
      if (pageableMemoryAccess) {
        printf("full unified memory support");
        if (pageableMemoryAccessUsesHostPageTables) {
          printf(" with hardware coherency\n");
        } else {
          printf(" with software coherency\n");
        }
      } else {
        printf(
            "full unified memory support for CUDA-made managed,allocations\n");
      }
    } else {
      printf("limited unified memory support: Windows, WSL, or Tegra\n");
    }
  }
}

/**
 * @brief Main function
 */
int main() {
  printf("cuda_graphs - CUDA implementation\n");
  printf("Chapter 3: CUDA Graphs exploration\n\n");

  float *input = nullptr;
  float *output = nullptr;

  // Allocate Pinned Host Memory
  CUDA_CHECK(cudaMallocHost((void **)&input, sizeof(float) * N));
  CUDA_CHECK(cudaMallocHost((void **)&output, sizeof(float) * N));

  // Initialize data so we can verify the math
  for (int i = 0; i < N; i++) {
    input[i] = static_cast<float>(i);
    output[i] = 0.0f;
  }

  // Run the operations
  cuda_graphs_pipelined(input, output);
  cuda_graphs_captured(input, output);

  // Print results to verify success
  print_array("Input", input, N);
  print_array("Output", output, N);
  // queryDevices();

  // Clean up host memory
  CUDA_CHECK(cudaFreeHost(input));
  CUDA_CHECK(cudaFreeHost(output));

  return 0;
}
