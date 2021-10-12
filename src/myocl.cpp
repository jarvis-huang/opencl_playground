#include "opencl_playground/myocl.hpp"

cl::Platform getDefaultPlatform() {
  // get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  checkErr(all_platforms.size() != 0 ? CL_SUCCESS : -1,
           "No platforms found. Check OpenCL installation!");
  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
  return default_platform;
}

cl::Device getDefaultDevice(cl::Platform default_platform) {
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
  return default_device;
}

cl::Program makeProgramFromKernelCode(const char* filename,
                                      cl::Context context) {
  // read kernel code from disk, convert it to a string
  std::ifstream f_kernel(filename);
  checkErr(f_kernel.is_open() ? CL_SUCCESS : -1, filename);
  std::string kernel_code(std::istreambuf_iterator<char>(f_kernel),
                          (std::istreambuf_iterator<char>()));

  cl_int err;
  cl::Program program(context, kernel_code, true,
                      &err);  // builds (compiles and links) the kernel code

  if (err != CL_SUCCESS) {
    std::cout << " Error building: " << filename << "\n";
    exit(1);
  }
  return program;
}
