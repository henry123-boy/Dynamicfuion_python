function(NNRT_add_test name)
    if(${ARGC} EQUAL 1)
        add_executable(${name} ${name}.cpp)
    else()
        add_executable(${name} ${ARGN})
    endif()
    target_link_libraries(${name} PUBLIC nnrt_cpp test_utils)
    target_link_libraries(${name} PUBLIC Python3::Python ${CATCH2_TARGET})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

if (BUILD_CPP_TESTS)
    ### test utilities library ###

    set(STATIC_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_data/)
    set(GENERATED_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data/)
    file(MAKE_DIRECTORY ${GENERATED_TEST_DATA_DIRECTORY})
    configure_file(test_utils/test_data_paths.hpp.in test_utils/test_data_paths.hpp @ONLY)

    add_library(
        test_utils STATIC
        test_utils/test_utils.hpp
        test_utils/test_utils.tpp
        test_utils/test_utils.cpp
        test_utils/compile_time_string_concatenation.hpp
        test_utils/numpy_image.h
        test_utils/image_io/load_png.h
        test_utils/image_io/load_png.cpp
        test_utils/pixel.h
        test_utils/image_view.h
        test_utils/image_io/dng_shared.h
        test_utils/image_io/load_dng.h
        test_utils/image_io/load_dng.cpp
        test_utils/enable_if.h
        test_utils/entropy_range_decoder.h
        test_utils/entropy_range_decoder.cpp
        test_utils/uintn.h
        test_utils/entropy_decoder_model.h
        test_utils/float_details.h
        test_utils/vectorstream.h
        test_utils/algs.h
        test_utils/image_io/save_png.h
        test_utils/image_io/save_png.cpp
        test_utils/array2d.h
        test_utils/image_io/save_dng.h
        test_utils/entropy_encoder_model.h
        test_utils/entropy_range_encoder.h
        test_utils/entropy_range_encoder.cpp
        test_utils/geometry.cpp
        test_utils/geometry.h)

    target_link_libraries(test_utils PUBLIC NNRT::3rdparty_libpng Python3::Module nnrt_cpp NNRT::3rdparty_fmt)
    target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/test_utils)


    ### add tests ###
    NNRT_add_test(test_graph_proc)
    NNRT_add_test(test_index_conversions)
    NNRT_add_test(test_array_comparisons)
    NNRT_add_test(test_heap)
    NNRT_add_test(test_kdtree)
    NNRT_add_test(test_graph_warp_field)
    NNRT_add_test(test_mesh_warping)
    NNRT_add_test(test_downsampling)
    NNRT_add_test(test_non_rigid_surface_voxel_block_grid)
    NNRT_add_test(test_matmul3d)
    NNRT_add_test(test_normals_operations)
    NNRT_add_test(test_tensor_io)
    NNRT_add_test(test_voxel_block_grid_io)
    NNRT_add_test(test_unproject_3d_points)
    NNRT_add_test(test_extract_face_vertices)

endif ()