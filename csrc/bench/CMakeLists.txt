set(BENCHMARK_NAME kdtree_benchmark)

add_executable(${BENCHMARK_NAME} kdtree_benchmark.cpp)
target_link_libraries(${BENCHMARK_NAME} PUBLIC nnrt_cpp)