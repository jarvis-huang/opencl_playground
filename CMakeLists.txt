cmake_minimum_required(VERSION 3.0.0)
project(opencl_playground VERSION 0.1.0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations -O3")

find_package(OpenCL REQUIRED)
find_package(Boost REQUIRED)

include_directories(
    # Better Flow include files
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    # GPU acceleration
    ${OpenCL_INCLUDE_DIRS}

    ${Boost_INCLUDE_DIRS}
)

add_executable(vector_add src/vector_add.cpp src/myocl.cpp)
target_link_libraries(vector_add ${OpenCL_LIBRARY})

add_executable(matrix_mul src/matrix_mul.cpp src/myocl.cpp)
target_link_libraries(matrix_mul ${OpenCL_LIBRARY})

add_executable(compute_pi src/compute_pi.cpp src/myocl.cpp)
target_link_libraries(compute_pi ${OpenCL_LIBRARY})

#enable_testing()