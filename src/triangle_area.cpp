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
  int ORDER = 12800; // WG_size=128, num_WG=100
  int WG_size = 128*4; // prefer multiples of 128
  int num_wg = ORDER / WG_size;
  int NREPEAT = 500;

  // Timers
  util::Timer timer_seq, timer_opencl;

  // Prepare input data
  srand (0); /* initialize random seed: */
  std::vector<float> v_left(NREPEAT), v_up(NREPEAT), v_right(NREPEAT), v_gt(NREPEAT);
  for (int i = 0; i < NREPEAT; i++) {
    v_left[i] = float(rand() % 1000 + 1) / 20.0;
    v_up[i] = float(rand() % 1000 + 1) / 20.0;
    v_right[i] = float(rand() % 1000 + 1) / 20.0;
    v_gt[i] = (v_left[i]+v_right[i])*v_up[i]/2.0;
  }

  // Prepare host and device memory
  timer_opencl.Tic();
  std::vector<float> res_opencl, res_seq;
  std::vector<float> h_C(num_wg);  // Host memory for temporary result
  cl::Buffer d_c;  // Device memory
  d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_wg);
  cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * WG_size);
  
  cl::NDRange global(ORDER);
  cl::NDRange local(WG_size);
  cl::make_kernel<cl::Buffer, cl::LocalSpaceArg, float, float, float> triangle_area(program, "triangle_area"); // Avoid this in a loop since it consumes significant run-time!

  for (int n = 0; n < NREPEAT; n++) {
    // run the kernel
    triangle_area(cl::EnqueueArgs(queue, global, local), d_c, localmem, v_left[n], v_up[n], v_right[n]).wait();

    // read result C from the device to array C
    cl::copy(queue, d_c, h_C.begin(), h_C.end());
    float suma = 0;
    for (auto x: h_C) suma += x;
    res_opencl.push_back(suma);
  }
  timer_opencl.Toc();

  srand (0);
  timer_seq.Tic();
  for (int n = 0; n < NREPEAT; n++) {
    float left = v_left[n];
    float up = v_up[n];
    float right = v_right[n];
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
  std::cout << "\n----------------------------------------------\n";
  std::cout << "    result (GT): ";
  std::cout << boost::format("%.3f %.3f %.3f") % v_gt[0] % v_gt[10] % v_gt[20];
  std::cout << "\n";

  // Time profiling
  std::cout << boost::format("Time per run (ms): %.3f (opencl) %.3f (seq)\n") %
                   (timer_opencl.ElapsedMs()/float(NREPEAT)) % 
                   (timer_seq.ElapsedMs()/float(NREPEAT));
  return 0;
}