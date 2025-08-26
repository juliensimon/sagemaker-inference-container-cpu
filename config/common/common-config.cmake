# Common configuration for all architectures

# Build settings
set(LLAMA_BUILD_SERVER ON)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_BENCHMARKS OFF)
set(BUILD_SHARED_LIBS OFF)

# Common compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Common optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Common definitions
add_definitions(-DGGML_USE_ACCELERATE)
add_definitions(-DGGML_USE_OPENBLAS)

# Common include directories
include_directories(${CMAKE_SOURCE_DIR})

# Common library settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common threading settings
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")

# Common memory settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_MAX_PARAMS=2048")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_MAX_CONTEXTS=64")
