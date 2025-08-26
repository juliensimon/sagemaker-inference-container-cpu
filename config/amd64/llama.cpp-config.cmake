# AMD64/Intel-specific optimizations for llama.cpp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")