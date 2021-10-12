#include <CL/cl.hpp>
#include <boost/format.hpp>
#include <cmath>     // sqrt
#include <fstream>   // file I/O
#include <iostream>  // printing

#include "opencl_playground/myocl.hpp"

#define __CL_ENABLE_EXCEPTIONS

double compute_pi_sequential(int num_steps) {
  double suma = 0.0;
  double step = 1.0 / num_steps;
#if 1
  for (double x = step / 2.0; x < 1.0; x += step) {
    double y = 4.0 / (1 + x * x);
    suma += y;
  }
#else
  for (double x = step / 2.0; x < 2.0; x += step) {
    double y = std::sqrt(4 - x * x);
    suma += y;
  }
#endif
  double pi = suma * step;
  return pi;
}

int main() {
  // Timers
  util::Timer timer_seq, timer_ocl;

  cl::Platform default_platform = util::getDefaultPlatform();
  cl::Device default_device = util::getDefaultDevice(default_platform);
  cl::Context context({default_device});
  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);
  cl::Program program =
      util::makeProgramFromKernelCode("../src/compute_pi.cl", context);

  // create buffers on the device
  int NREPEAT = 100;
  int SIZE = 800000;
  int global_size = 10000;
  int local_size = 100;
  int n_groups = global_size / local_size;
  float step = float(1.0) / SIZE;
  int ninput_per_group = int(SIZE / n_groups);

  // Prepare host and device memory
  std::vector<float> h_A(n_groups);  // Host memory for Matrix A
  cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * local_size);

  // run the kernel
  timer_ocl.Tic();
  cl::make_kernel<cl::Buffer, cl::LocalSpaceArg, float, int> compute_pi(
      program, "compute_pi");
  cl::NDRange global(global_size);
  cl::NDRange local(local_size);
  double pi_opencl = 0.0;
  for (int n = 0; n < NREPEAT; n++) {
    cl::Buffer d_a =
        cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n_groups);
    compute_pi(cl::EnqueueArgs(queue, global, local), d_a, localmem, step,
               ninput_per_group)
        .wait();

    // read result C from the device to array C
    cl::copy(queue, d_a, h_A.begin(), h_A.end());
    double pi_opencl_this = 0.0;
    for (auto& x : h_A) pi_opencl_this += x;
    pi_opencl += pi_opencl_this / SIZE;
  }
  pi_opencl /= NREPEAT;
  timer_ocl.Toc();
  std::cout << boost::format("ocl: err=%.2e, time=%.0f us\n") %
                   (pi_opencl - M_PI) %
                   (double(timer_ocl.ElapsedUs()) / NREPEAT);

  timer_seq.Tic();
  double pi_seq = 0.0;
  for (int n = 0; n < NREPEAT; n++) {
    pi_seq += compute_pi_sequential(SIZE);
  }
  pi_seq /= NREPEAT;
  timer_seq.Toc();
  std::cout << boost::format("seq: err=%.2e, time=%.0f us\n") %
                   (pi_seq - M_PI) % (double(timer_seq.ElapsedUs()) / NREPEAT);

  // Time profiling
  // std::cout << boost::format("Time (ms): %d (opencl) %d (seq)\n") %
  //                  timer_ocl.ElapsedMs() % timer_seq.ElapsedMs();

  float dSeconds_cl = timer_ocl.ElapsedSec();
  float dSeconds_seq = timer_seq.ElapsedSec();
  float dNumOps = (double)SIZE;
  float gflops_cl = 1.0e-9 * dNumOps / dSeconds_cl;
  float gflops_seq = 1.0e-9 * dNumOps / dSeconds_seq;
  // std::cout << boost::format("GFLOPS: %.1f (opencl) %.1f (seq)\n") %
  // gflops_cl %
  //                 gflops_seq;

  return 0;
}