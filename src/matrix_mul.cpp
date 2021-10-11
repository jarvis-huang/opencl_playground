#include <CL/cl.hpp>
#include <boost/format.hpp>
#include <chrono>    // for time measurement
#include <fstream>   // for file I/O
#include <iostream>  // for printing

#include "opencl_playground/myocl.hpp"

using namespace std::chrono;
constexpr auto time_now = std::chrono::high_resolution_clock::now;

void matrix_mul_sequential(const int* A, const int* B, int* C, const int N) {
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
  cl::Platform default_platform = getDefaultPlatform();
  cl::Device default_device = getDefaultDevice(default_platform);
  cl::Context context({default_device});
  cl::Program program =
      makeProgramFromKernelCode("../src/matrix_mul.cl", context);

  // create buffers on the device
  int N = 500;
  int N2 = N * N;
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * N2);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * N2);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * N2);

  // Prepare input data
  int* A = new int[N2];
  int* B = new int[N2];
  for (int i = 0; i < N2; i++) {
    A[i] = i % 10 - 2;
    B[i] = i % 7 - 4;
  }

  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  // write arrays A and B to the device
  auto t0 = time_now();
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * N2, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * N2, B);

  // run the kernel
  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> matrix_mul(
      program, "matrix_mul");
  cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(N), cl::NullRange);
  auto t_start = time_now();
  matrix_mul(eargs, buffer_A, buffer_B, buffer_C, N).wait();
  auto t_end = time_now();
  auto dur_opencl = duration_cast<milliseconds>(t_end - t_start).count();

  // read result C from the device to array C
  int* C = new int[N2];
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * N2, C);
  auto t1 = time_now();
  auto dur_opencl_mem = duration_cast<milliseconds>(t1 - t0).count();

  t_start = time_now();
  int* C_seq = new int[N2];
  matrix_mul_sequential(A, B, C_seq, N);
  t_end = time_now();
  auto dur_seq = duration_cast<milliseconds>(t_end - t_start).count();

  std::cout << " result (OpenCL): ";
  for (int i = 0; i < 10; i++) {
    std::cout << boost::format("%d ") % C[i];
  }
  std::cout << "\n";
  std::cout << " result (sequential): ";
  for (int i = 0; i < 10; i++) {
    std::cout << boost::format("%d ") % C_seq[i];
  }
  std::cout << "\n";

  // Time profiling
  std::cout << boost::format("Time opencl: %d ms (with mem: %d ms)\n") %
                   dur_opencl % dur_opencl_mem;
  std::cout << boost::format("Time seq: %d ms \n") % dur_seq;

  float dSeconds_cl = float(dur_opencl) / 1000.0;
  float dSeconds_seq = float(dur_seq) / 1000.0;
  float dNumOps = 2.0 * (double)(N * N * N);
  float gflops_cl = 1.0e-9 * dNumOps / dSeconds_cl;
  float gflops_seq = 1.0e-9 * dNumOps / dSeconds_seq;
  std::cout << boost::format("GFLOPS: %.2f (opencl) %.2f (seq)\n") % gflops_cl %
                   gflops_seq;
  return 0;
}