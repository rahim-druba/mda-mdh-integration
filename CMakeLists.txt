cmake_minimum_required(VERSION 2.8.11)
project(mda_mdh_integration)

set(CMAKE_CXX_STANDARD 14)

include_directories(mdh_framework)
include_directories(mdh_framework/include)

file(GLOB FRAMEWORK_SRC mdh_framework/src/*.cpp)
add_library(mdh_cuda_generator STATIC ${FRAMEWORK_SRC})

# Built from the bridge's output, not hand-written.
add_executable(poisson_from_uml generated/poisson_from_uml.cpp)
target_link_libraries(poisson_from_uml mdh_cuda_generator)

# The original hand-written MDH spec (copied unmodified from
# mdh-cuda-stencil-kernels/kernels/poisson/spec/poisson.cpp), built here for a
# direct side-by-side diff against the bridge's output.
add_executable(poisson_handwritten mdh_reference/poisson_handwritten.cpp)
target_link_libraries(poisson_handwritten mdh_cuda_generator)
