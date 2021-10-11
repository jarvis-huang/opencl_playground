#ifndef MYOCL_H
#define MYOCL_H

#include <CL/cl.hpp>
#include <fstream>   // for file I/O
#include <iostream>  // for printing

inline void checkErr(cl_int err, const char* name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

cl::Platform getDefaultPlatform();
cl::Device getDefaultDevice(cl::Platform default_platform);
cl::Program makeProgramFromKernelCode(const char* filename,
                                      cl::Context context);

#endif  // MYOCL_H