#include <CL/cl.hpp>
#include <chrono>  // for time measurement
//#include <cstdio>
//#include <cstdlib>
#include <fstream>   // for file I/O
#include <iostream>  // for printing
//#include <utility>

using namespace std::chrono;
constexpr auto time_now = std::chrono::high_resolution_clock::now;

inline void checkErr(cl_int err, const char* name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  // get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  checkErr(all_platforms.size() != 0 ? CL_SUCCESS : -1,
           "No platforms found. Check OpenCL installation!");
  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  // get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  checkErr(all_devices.size() != 0 ? CL_SUCCESS : -1,
           "No devices found. Check OpenCL installation!");
  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()
            << "\n";
  cl::Context context({default_device});

  // read kernel code from disk, convert it to a string
  std::ifstream f_kernel("../src/vector_add.cl");
  checkErr(f_kernel.is_open() ? CL_SUCCESS : -1, "vector_add.cl");
  std::string kernel_code(std::istreambuf_iterator<char>(f_kernel),
                          (std::istreambuf_iterator<char>()));
  cl::Program::Sources sources(
      1, std::make_pair(kernel_code.c_str(), kernel_code.length() + 1));

  cl::Program program(context, sources);
  if (program.build({default_device}) != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
              << "\n";
    exit(1);
  }

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