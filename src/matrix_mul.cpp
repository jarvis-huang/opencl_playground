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

void matrix_mul_sequential(const int* A, const int* B, int* C, int N) {
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
  auto max_work_items = default_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()
            << " (" << max_work_items[0] << ", " << max_work_items[1] << ", "
            << max_work_items[2] << ")\n";
  cl::Context context({default_device});

  // read kernel code from disk, convert it to a string
  std::ifstream f_kernel("../src/matrix_mul.cl");
  checkErr(f_kernel.is_open() ? CL_SUCCESS : -1, "matrix_mul.cl");
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
  int N = 700;
  int N2 = N * N;
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * N2);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * N2);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * N2);

  // Prepare input data
  // int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
  int* A = new int[N2];
  int* B = new int[N2];
  for (int i = 0; i < N2; i++) {
    A[i] = i % 10 - 2;
    B[i] = i % 7 - 4;
  }

  // create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  // write arrays A and B to the device
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * N2, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * N2, B);

  // run the kernel
  cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> matrix_mul(
      cl::Kernel(program, "matrix_mul"));
  cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(N), cl::NullRange);
  auto t_start = time_now();
  matrix_mul(eargs, buffer_A, buffer_B, buffer_C, N).wait();
  auto t_end = time_now();
  auto dur_opencl = duration_cast<milliseconds>(t_end - t_start).count();

  // read result C from the device to array C
  int* C = new int[N2];
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * N2, C);

  t_start = time_now();
  int* C_seq = new int[N2];
  matrix_mul_sequential(A, B, C_seq, N);
  t_end = time_now();
  auto dur_seq = duration_cast<milliseconds>(t_end - t_start).count();

  std::cout << " result (OpenCL): \n";
  for (int i = 0; i < 10; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << "\n";
  std::cout << " result (sequential): \n";
  for (int i = 0; i < 10; i++) {
    std::cout << C_seq[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Time opencl: " << dur_opencl << " ms" << std::endl;
  std::cout << "Time seq " << dur_seq << " ms" << std::endl;

  return 0;
}