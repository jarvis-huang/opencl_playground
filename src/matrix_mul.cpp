#include <CL/cl.hpp>
#include <boost/format.hpp>
#include <fstream>   // for file I/O
#include <iostream>  // for printing

#include "opencl_playground/myocl.hpp"

void matrix_mul_sequential(const std::vector<int>& A, const std::vector<int>& B,
                           std::vector<int>& C, const int N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N;
         j++) {  // loop over each location of output matrix C
      int s = 0;
      for (size_t k = 0; k < N;
           k++) {  // inner loop to compute vector dot product
        // A[i, k] * B[k, j]
        s += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = s;
    }
  }
}

int main() {
  cl::Platform default_platform = util::getDefaultPlatform();
  cl::Device default_device = util::getDefaultDevice(default_platform);
  cl::Context context({default_device});
  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);
  cl::Program program =
      util::makeProgramFromKernelCode("../src/matrix_mul.cl", context);
  // return 0;

  // create buffers on the device
  int ORDER = 500;
  int size = ORDER * ORDER;

  // Prepare host and device memory
  std::vector<int> h_A(size);  // Host memory for Matrix A
  std::vector<int> h_B(size);  // Host memory for Matrix B
  std::vector<int> h_C(size);  // Host memory for Matrix C
  for (int i = 0; i < size; i++) {
    h_A[i] = i % 19 - 8;
    h_B[i] = i % 37 - 5;
  }
  cl::Buffer d_a, d_b, d_c;  // Matrices in device memory
  d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);
  d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);
  d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * size);

  // Timers
  util::Timer timer_seq, timer_opencl;

  // run the kernel
  timer_opencl.Tic();
  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> matrix_mul(
      program, "matrix_mul");
  cl::NDRange global(ORDER);
  matrix_mul(cl::EnqueueArgs(queue, global), d_a, d_b, d_c, ORDER).wait();
  timer_opencl.Toc();

  // read result C from the device to array C
  cl::copy(queue, d_c, h_C.begin(), h_C.end());

  timer_seq.Tic();
  int* C_seq = new int[size];
  std::vector<int> h_C_seq(size);
  matrix_mul_sequential(h_A, h_B, h_C_seq, ORDER);
  timer_seq.Toc();

  std::cout << "result (OpenCL): ";
  for (int i = 0; i < 80; i += 8) {
    std::cout << boost::format("%d ") % h_C[i];
  }
  std::cout << "\n";
  std::cout << "   result (seq): ";
  for (int i = 0; i < 80; i += 8) {
    std::cout << boost::format("%d ") % h_C_seq[i];
  }
  std::cout << "\n";

  // Time profiling
  std::cout << boost::format("Time (ms): %d (opencl) %d (seq)\n") %
                   timer_opencl.ElapsedMs() % timer_seq.ElapsedMs();

  float dSeconds_cl = timer_opencl.ElapsedSec();
  float dSeconds_seq = timer_seq.ElapsedSec();
  float dNumOps = 2.0 * (double)(ORDER * ORDER * ORDER);
  float gflops_cl = 1.0e-9 * dNumOps / dSeconds_cl;
  float gflops_seq = 1.0e-9 * dNumOps / dSeconds_seq;
  std::cout << boost::format("GFLOPS: %.1f (opencl) %.1f (seq)\n") % gflops_cl %
                   gflops_seq;
  return 0;
}