#include <CL/cl.hpp>
#include <chrono>    // for time measurement
#include <fstream>   // for file I/O
#include <iostream>  // for printing

#include "opencl_playground/myocl.hpp"

using namespace std::chrono;
constexpr auto time_now = std::chrono::high_resolution_clock::now;

int main() {
  cl::Device default_device = util::getDevice("GPU");
  cl::Context context({default_device});
  cl::Program program =
      util::makeProgramFromKernelCode("../src/vector_add.cl", context);

  // create buffers on the device
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

  int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  // write arrays A and B to the device
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);

  // run the kernel
  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> vector_add(
      cl::Kernel(program, "vector_add"));
  cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
  auto start_time = time_now();
  vector_add(eargs, buffer_A, buffer_B, buffer_C).wait();
  auto end_time = time_now();
  auto duration = duration_cast<microseconds>(end_time - start_time).count();
  std::cout << "Time opencl: " << duration << " us" << std::endl;

  int C[10];
  // read result C from the device to array C
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

  std::cout << " result: \n";
  for (int i = 0; i < 10; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << "\n";

  return 0;
}