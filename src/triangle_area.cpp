#include <CL/cl.hpp>
#include <boost/format.hpp>
#include <fstream>   // for file I/O
#include <iostream>  // for printing
#include <stdlib.h>  // srand, rand
#include <time.h>    // time

#include "opencl_playground/myocl.hpp"

void triangle_area_sequential(std::vector<float>& C, const float left, const float up, const float right) {
  int N = C.size();
  for (size_t i = 0; i < N; i++) {
    float step = up / N;
    float y = i * step;
    float bound_left = (up - y) / up * left;
    float bound_right = (up - y) / up * right;
    C[i] = (bound_left + bound_right) * step;
  }
  float sum = 0;
  for (size_t i = 0; i < N; i++) {
    sum += C[i];
  }
  C[0] = sum;
}

int main() {
  cl::Device default_device = util::getDevice("CPU");
  cl::Context context({default_device});
  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);
  cl::Program program =
      util::makeProgramFromKernelCode("../src/triangle_area.cl", context);

  // create buffers on the device
  int ORDER = 2000;
  int NREPEAT = 1000;

  // Timers
  util::Timer timer_seq, timer_opencl;

  // Prepare host and device memory
  /* initialize random seed: */
  srand (0);
  timer_opencl.Tic();
  std::vector<float> res_opencl, res_seq, res_gt;
  for (int n = 0; n < NREPEAT; n++) {
    float left = float(rand() % 1000 + 1) / 20.0;
    float up = float(rand() % 1000 + 1) / 20.0;
    float right = float(rand() % 1000 + 1) / 20.0;

    std::vector<float> h_C(ORDER);  // Host memory for temporary result
    cl::Buffer d_c;  // Device memory
    d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * ORDER);

    // run the kernel
    cl::make_kernel<cl::Buffer, float, float, float> triangle_area(program, "triangle_area");
    cl::NDRange global(ORDER);
    cl::NDRange local(ORDER);
    triangle_area(cl::EnqueueArgs(queue, global, local), d_c, left, up, right).wait();

    // read result C from the device to array C
    cl::copy(queue, d_c, h_C.begin(), h_C.end());
    res_opencl.push_back(h_C[0]);
    res_gt.push_back((left+right)*up/2.0);
  }
  timer_opencl.Toc();

  srand (0);
  timer_seq.Tic();
  for (int n = 0; n < NREPEAT; n++) {
    float left = float(rand() % 1000 + 1) / 20.0;
    float up = float(rand() % 1000 + 1) / 20.0;
    float right = float(rand() % 1000 + 1) / 20.0;
    std::vector<float> h_C_seq(ORDER);
    triangle_area_sequential(h_C_seq, left, up, right);
    res_seq.push_back(h_C_seq[0]);
  }
  timer_seq.Toc();

  std::cout << "result (OpenCL): ";
  std::cout << boost::format("%.3f %.3f %.3f") % res_opencl[0] % res_opencl[10] % res_opencl[20];
  std::cout << "\n";
  std::cout << "   result (seq): ";
  std::cout << boost::format("%.3f %.3f %.3f") % res_seq[0] % res_seq[10] % res_seq[20];
  std::cout << "\n";
  std::cout << "    result (GT): ";
  std::cout << boost::format("%.3f %.3f %.3f") % res_gt[0] % res_gt[10] % res_gt[20];
  std::cout << "\n";

  // Time profiling
  std::cout << boost::format("Time (ms): %d (opencl) %d (seq)\n") %
                   timer_opencl.ElapsedMs() % timer_seq.ElapsedMs();

  // float dSeconds_cl = timer_opencl.ElapsedSec();
  // float dSeconds_seq = timer_seq.ElapsedSec();
  // float dNumOps = 2.0 * (double)(ORDER * ORDER * ORDER);
  // float gflops_cl = 1.0e-9 * dNumOps / dSeconds_cl;
  // float gflops_seq = 1.0e-9 * dNumOps / dSeconds_seq;
  // std::cout << boost::format("GFLOPS: %.1f (opencl) %.1f (seq)\n") % gflops_cl %
  //                  gflops_seq;
  return 0;
}