function(add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PUBLIC nnrt_cpp test_utils)
    target_link_libraries(${name} PUBLIC Python3::Python ${CATCH2_TARGET})
endfunction()

if (BUILD_CPP_TESTS)
    ### test utilities library ###

    set(STATIC_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_data/)
    set(GENERATED_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data/)
    file(MAKE_DIRECTORY ${GENERATED_TEST_DATA_DIRECTORY})
    configure_file(test_utils/test_data_paths.hpp.in test_utils/test_data_paths.hpp @ONLY)

    add_library(test_utils STATIC
        test_utils/test_utils.hpp
        test_utils/test_utils.tpp
        test_utils/test_utils.cpp
        test_utils/compile_time_string_concatenation.hpp
        test_utils/numpy_image.h
        test_utils/pixel.h
        test_utils/algs.h
        test_utils/enable_if.h
        test_utils/uintn.h
        test_utils/serialize.h
        test_utils/noncopyable.h
        test_utils/error.h
        test_utils/float_details.h
        test_utils/matrix.h
        test_utils/is_kind.h
        test_utils/memory_manager.h
        test_utils/stl_checked.h
        test_utils/std_allocator.h
        test_utils/matrix/matrix_utilities.h
        test_utils/array.h
        test_utils/array2d.h
        test_utils/image_processing/generic_image.h
        test_utils/sort.h
        test_utils/memory_manager_stateless.h
        test_utils/vectorstream.h
        test_utils/byte_orderer.h
        test_utils/image_loader/png_loader.h
        test_utils/image_loader/png_loader.cpp
        test_utils/image_loader/png_loader_abstract.h)

    target_link_libraries(test_utils PUBLIC Python3::Module nnrt_cpp)
    target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/test_utils)


    ### add tests ###
    add_test(test_graph_proc test_graph_proc.cpp)
    add_test(test_index_conversions test_index_conversions.cpp)
    add_test(test_array_comparisons test_array_comparisons.cpp)
    add_test(test_experiments test_experiments.cpp)
    add_test(test_heap test_heap.cpp)
endif ()